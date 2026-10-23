from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from hdfs import InsecureClient
import json
import asyncio
import logging
import os
from datetime import datetime
from pydantic import BaseModel
from dqn_engine import dqn_recommender

try:
    import pymongo
except ImportError:
    pymongo = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("api_usages")

app = FastAPI(
    title="API d'exposition des Usages TELCO/OMY depuis HDFS & MongoDB Atlas",
    description="API haute disponibilité connectée directement à la base MongoDB Atlas du tracking-service",
    version="1.4.0"
)

# Configuration CORS pour Flutter et Spring Boot
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dictionnaire pour stocker les profils en cache mémoire
PROFILES_CACHE = {}

# Connexion MongoDB Atlas (Même cluster que tracking-service)
MONGO_URI = os.getenv(
    "SPRING_DATA_MONGODB_URI",
    "your_mongo_url"
)

# Fichier de persistance locale de secours
BEHAVIORAL_CLICKS_FILE = "behavioral_clicks.json"

MONGO_CLIENT = None
MONGO_DB = None
PROFILES_COLL = None
CLICKS_COLL = None

def get_mongo_collections():
    """Initialisation tardive et sécurisée de MongoDB Atlas pour éviter les delais DNS au demarrage"""
    global MONGO_CLIENT, MONGO_DB, PROFILES_COLL, CLICKS_COLL
    if PROFILES_COLL is not None:
        return PROFILES_COLL, CLICKS_COLL

    if pymongo is None:
        return None, None

    try:
        MONGO_CLIENT = pymongo.MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=1000,
            connectTimeoutMS=1000,
            readPreference='primaryPreferred'
        )
        MONGO_DB = MONGO_CLIENT.get_database("maxitprototype")
        PROFILES_COLL = MONGO_DB["user_profiles"]
        CLICKS_COLL = MONGO_DB["behavioral_clicks"]
        logger.info("==================================================")
        logger.info("   BASE MONGODB ATLAS CONNECTÉE (maxitprototype)  ")
        logger.info("==================================================")
        return PROFILES_COLL, CLICKS_COLL
    except Exception as e:
        logger.debug("MongoDB Atlas non accessible (%s). Utilisation du cache local et mémoire.", e)
        return None, None

# Chemin HDFS où les fichiers JSON sont stockés
HDFS_DATA_PATH = '/api_output_json'
IS_LOADING_HDFS = False
IS_WARM_STARTED = False

def save_to_mongo(msisdn: str, profile_data: list):
    """Sauvegarde/met à jour le profil client dans MongoDB Atlas dans la collection 'user_profiles'"""
    profiles_coll, _ = get_mongo_collections()
    if profiles_coll is not None:
        try:
            profiles_coll.update_one(
                {"msisdn": msisdn},
                {"$set": {"msisdn": msisdn, "profiles": profile_data, "updatedAt": datetime.now().isoformat()}},
                upsert=True
            )
        except Exception as e:
            logger.error("Erreur d'écriture MongoDB pour %s : %s", msisdn, e)

def get_from_mongo(msisdn: str):
    """Récupère le profil client depuis la collection MongoDB Atlas 'user_profiles'"""
    profiles_coll, _ = get_mongo_collections()
    if profiles_coll is not None:
        try:
            doc = profiles_coll.find_one({"msisdn": msisdn})
            if doc and "profiles" in doc:
                return doc["profiles"]
        except Exception as e:
            logger.error("Erreur de lecture MongoDB pour %s : %s", msisdn, e)
    return None

def save_behavioral_click(msisdn: str, universe: str, mode: str, service_id: str, reward: float = 0.2):
    """Enregistre le clic/récompense dans MongoDB Atlas (collections 'behavioral_clicks' & 'tracking_events') et fichier local"""
    action_type = "TRANSACTION" if reward >= 1.0 else "CLICK"
    click_doc = {
        "msisdn": msisdn,
        "universe": universe,
        "mode": mode,
        "serviceId": service_id,
        "eventType": service_id,
        "actionType": action_type,
        "reward": reward,
        "sourceService": "personnalisation-service",
        "timestamp": datetime.now().isoformat()
    }
    _, clicks_coll = get_mongo_collections()
    if clicks_coll is not None:
        try:
            clicks_coll.insert_one(click_doc.copy())
        except Exception as e:
            logger.error("Erreur d'insertion du clic dans MongoDB : %s", e)

    click_doc.pop("_id", None)

    # Sauvegarde locale de secours
    clicks = []
    if os.path.exists(BEHAVIORAL_CLICKS_FILE):
        try:
            with open(BEHAVIORAL_CLICKS_FILE, "r", encoding="utf-8") as f:
                clicks = json.load(f)
        except Exception:
            clicks = []
    clicks.append(click_doc)
    try:
        with open(BEHAVIORAL_CLICKS_FILE, "w", encoding="utf-8") as f:
            json.dump(clicks, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("Erreur lors de la sauvegarde locale du clic : %s", e)

def restore_behavioral_clicks():
    """Restaure les clics comportementaux depuis MongoDB Atlas et le fichier local dans PROFILES_CACHE"""
    restored_count = 0
    all_clicks = []
    _, clicks_coll = get_mongo_collections()

    # 1. Lecture depuis MongoDB Atlas
    if clicks_coll is not None:
        try:
            docs = list(clicks_coll.find())
            for d in docs:
                all_clicks.append({
                    "msisdn": d.get("msisdn"),
                    "universe": d.get("universe", "TELCO"),
                    "mode": d.get("mode", "SIMPLE"),
                    "serviceId": d.get("serviceId"),
                    "timestamp": d.get("timestamp")
                })
        except Exception as e:
            logger.error("Erreur de lecture des clics MongoDB : %s", e)

    # 2. Lecture complémentaire du fichier local
    if os.path.exists(BEHAVIORAL_CLICKS_FILE):
        try:
            with open(BEHAVIORAL_CLICKS_FILE, "r", encoding="utf-8") as f:
                local_clicks = json.load(f)
                all_clicks.extend(local_clicks)
        except Exception:
            pass

    for click in all_clicks:
        msisdn = click.get("msisdn")
        universe = click.get("universe", "TELCO")
        mode = click.get("mode", "SIMPLE")
        service_id = click.get("serviceId")
        ts = click.get("timestamp")

        if not msisdn or not service_id:
            continue

        profiles = PROFILES_CACHE.get(msisdn, [])
        if not profiles:
            PROFILES_CACHE[msisdn] = [{
                "_source": {
                    "msisdn": msisdn,
                    "univers": universe,
                    "mode": mode,
                    "liste_de_services": [service_id],
                    "date": ts,
                    "last_date": ts
                }
            }]
        else:
            found = False
            for profile in profiles:
                src = profile.get("_source", {})
                if src.get("univers") == universe and src.get("mode") == mode:
                    services = src.get("liste_de_services", [])
                    services.append(service_id)
                    found = True
                    break
            if not found:
                profiles.append({
                    "_source": {
                        "msisdn": msisdn,
                        "univers": universe,
                        "mode": mode,
                        "liste_de_services": [service_id],
                        "date": ts,
                        "last_date": ts
                    }
                })
        restored_count += 1

    if restored_count > 0:
        logger.info("Restauration permanente réussie de %d clics comportementaux depuis MongoDB Atlas / Fichier local.", restored_count)

def load_profile():
    """Charge les données HDFS et fusionne les données MongoDB Atlas"""
    global PROFILES_CACHE, IS_LOADING_HDFS

    if IS_LOADING_HDFS:
        logger.warning("Un chargement HDFS est déjà en cours d'exécution. Requête ignorée pour éviter les accès simultanés.")
        return

    IS_LOADING_HDFS = True
    logger.info("==================================================")
    logger.info("   CHARGEMENT GLOBAL DES DONNÉES HDFS & MONGODB ATLAS  ")
    logger.info("==================================================")

    try:
        hdfs_client = InsecureClient('http://localhost:9870', user='dell')
        files = hdfs_client.list(HDFS_DATA_PATH)
        files_part = [f for f in files if f.startswith('part-')]

        temp_map = {}

        for fileName in files_part:
            hdfs_path = f"{HDFS_DATA_PATH}/{fileName}"
            logger.info("Reading file '%s/part-...'", HDFS_DATA_PATH)
            with hdfs_client.read(hdfs_path, encoding='utf-8') as reader:
                content = reader.read()
                for line in content.splitlines():
                    if line.strip():
                        data = json.loads(line)
                        msisdn = data.get("msisdn")
                        if not msisdn:
                            continue

                        univers = data.get("univers") or "TELCO"
                        mode = data.get("mode") or "SIMPLE"
                        raw_services = data.get("liste_de_services") or []
                        if isinstance(raw_services, str):
                            raw_services = [raw_services]

                        date_val = data.get("date")
                        last_date_val = data.get("last_date")

                        key = (univers, mode)
                        if msisdn not in temp_map:
                            temp_map[msisdn] = {}

                        if key not in temp_map[msisdn]:
                            temp_map[msisdn][key] = {
                                "msisdn": msisdn,
                                "univers": univers,
                                "mode": mode,
                                "services": set(raw_services),
                                "date": date_val,
                                "last_date": last_date_val
                            }
                        else:
                            entry = temp_map[msisdn][key]
                            entry["services"].update(raw_services)
                            if date_val and (not entry["date"] or date_val > entry["date"]):
                                entry["date"] = date_val
                            if last_date_val and (not entry["last_date"] or last_date_val > entry["last_date"]):
                                entry["last_date"] = last_date_val

        temp_cache = {}
        total_blocs = 0
        for msisdn, modes_map in temp_map.items():
            temp_cache[msisdn] = []
            for (univers, mode), entry in modes_map.items():
                bloc = {
                    "_source": {
                        "msisdn": entry["msisdn"],
                        "univers": entry["univers"],
                        "liste_de_services": list(entry["services"]),
                        "mode": entry["mode"],
                        "date": entry["date"],
                        "last_date": entry["last_date"]
                    }
                }
                temp_cache[msisdn].append(bloc)
            total_blocs += 1

        global IS_WARM_STARTED
        if temp_cache:
            for msisdn, new_blocs in temp_cache.items():
                if msisdn not in PROFILES_CACHE:
                    PROFILES_CACHE[msisdn] = new_blocs
                else:
                    existing_blocs = PROFILES_CACHE[msisdn]
                    for nb in new_blocs:
                        n_src = nb.get("_source", {})
                        u, m = n_src.get("univers"), n_src.get("mode")
                        found = False
                        for eb in existing_blocs:
                            e_src = eb.get("_source", {})
                            if e_src.get("univers") == u and e_src.get("mode") == m:
                                s_set = set(e_src.get("liste_de_services", []))
                                s_set.update(n_src.get("liste_de_services", []))
                                e_src["liste_de_services"] = list(s_set)
                                found = True
                                break
                        if not found:
                            existing_blocs.append(nb)

            logger.info("Mise à jour réussie du cache : %d clients (%d blocs consolidés)", len(PROFILES_CACHE), total_blocs)

            if not IS_WARM_STARTED:
                try:
                    dqn_recommender.warm_start_from_hdfs(PROFILES_CACHE)
                    IS_WARM_STARTED = True
                except Exception as ex_dqn:
                    logger.error("Erreur lors du Warm-Start DQN : %s", ex_dqn)
        else:
            logger.warning("Aucun nouveau profil trouvé sur HDFS. Conservation du cache actif (%d clients).", len(PROFILES_CACHE))

        # Restauration permanente des clics comportementaux enregistrés par l'API
        restore_behavioral_clicks()

    except Exception as e:
        logger.error("Erreur lors de la lecture HDFS : %s", e)
    finally:
        IS_LOADING_HDFS = False

async def auto_refresh_hdfs_task():
    """Tâche d'arrière-plan pour synchroniser le cache avec Spark Streaming toutes les 30 minutes"""
    while True:
        await asyncio.sleep(1800)
        load_profile()

# Model DTO pour les récompenses DQN
class RewardRequest(BaseModel):
    msisdn: str
    serviceId: str
    reward: float
    universe: str = "TELCO"
    mode: str = "SIMPLE"

# Démarrage de l'api
@app.on_event("startup")
async def startup_event():
    load_profile()
    restore_behavioral_clicks()
    asyncio.create_task(auto_refresh_hdfs_task())

# Endpoints Deep Q-Network (DQN)
@app.get("/api/v1/recommendations/{msisdn}")
def get_dqn_recommendations(msisdn: str, univers: str = "TELCO", mode: str = "SIMPLE"):
    """Retourne la liste des recommandations générées par le réseau de neurones PyTorch DQN + Personnalisation par numéro"""
    # === TPS-01 & TPS-02 : RÉCUPÉRATION DES RECOMMANDATIONS & IDENTIFICATION PAR MSISDN ===
    profiles = PROFILES_CACHE.get(msisdn)
    if not profiles:
        mongo_p = get_from_mongo(msisdn)
        if mongo_p:
            PROFILES_CACHE[msisdn] = mongo_p
            profiles = mongo_p

    # === TPS-09 : GESTION DES ERREURS (EXCEPTION HTTP 404 SI NUMÉRO INCONNU) ===
    if not profiles:
        raise HTTPException(status_code=404, detail=f"Client {msisdn} introuvable")

    telco_c = 0
    omy_c = 0
    user_counts = {}
    for profile in profiles:
        src = profile.get("_source", {})
        if src.get("univers") == "TELCO":
            telco_c += len(src.get("liste_de_services", []))
        elif src.get("univers") == "OMY":
            omy_c += len(src.get("liste_de_services", []))
        
        for srv in src.get("liste_de_services", []):
            user_counts[srv] = user_counts.get(srv, 0) + 1

    current_hour = datetime.now().hour

    state = dqn_recommender.build_state_vector(
        telco_count=telco_c,
        omy_count=omy_c,
        hour=current_hour,
        mode=mode,
        universe=univers
    )

    # === TPS-04 & TPS-05 : CLASSEMENT PAR SCORE ET GESTION DES DOUBLONS ===
    recommendations = dqn_recommender.predict_recommendations(
        state, 
        univers=univers, 
        top_n=10, 
        user_counts=user_counts
    )
    
    # === TPS-03 : STRUCTURE DE LA RÉPONSE API STANDARDISÉE POUR FLUTTER ===
    return {
        "msisdn": msisdn,
        "univers": univers,
        "mode": mode,
        "algorithm": "Deep Q-Network (PyTorch) + Hybrid Personalization",
        "recommendations": [
            {"serviceId": srv, "qValue": round(float(score), 4)}
            for srv, score in recommendations
        ]
    }

@app.post("/api/v1/recommendations/reward")
def record_dqn_reward(req: RewardRequest):
    """Enregistre une récompense (R=+0.2 au clic, R=+1.0 à l'achat) et persiste le profil dans MongoDB Atlas"""
    # === TPS-06 & TPS-07 : ENVOI DU FEEDBACK ET PRISE EN COMPTE DU REWARD ===
    profiles = PROFILES_CACHE.get(req.msisdn)
    if not profiles:
        mongo_p = get_from_mongo(req.msisdn)
        if mongo_p:
            PROFILES_CACHE[req.msisdn] = mongo_p
            profiles = mongo_p

    # === TPS-09 : GESTION DES ERREURS (EXCEPTION HTTP 404 SI NUMÉRO INCONNU) ===
    if not profiles:
        raise HTTPException(status_code=404, detail=f"Client {req.msisdn} introuvable")

    telco_c = 0
    omy_c = 0
    for profile in profiles:
        src = profile.get("_source", {})
        if src.get("univers") == "TELCO":
            telco_c += len(src.get("liste_de_services", []))
        elif src.get("univers") == "OMY":
            omy_c += len(src.get("liste_de_services", []))

    current_hour = datetime.now().hour

    state = dqn_recommender.build_state_vector(
        telco_count=telco_c,
        omy_count=omy_c,
        hour=current_hour,
        mode=req.mode,
        universe=req.universe
    )

    dqn_recommender.record_feedback(state, req.serviceId, req.reward)
    logger.info("[DQN REWARD] MSISDN %s -> Service %s : Reward %f enregistré avec succès.", req.msisdn, req.serviceId, req.reward)

    # 1. Mise à jour du cache mémoire (Accumulation de chaque clic pour rehausser le score)
    found = False
    for profile in profiles:
        src = profile.get("_source", {})
        if src.get("univers") == req.universe and src.get("mode") == req.mode:
            services = src.get("liste_de_services", [])
            services.append(req.serviceId)
            src["last_date"] = datetime.now().isoformat()
            found = True
            break
    if not found:
        profiles.append({
            "_source": {
                "msisdn": req.msisdn,
                "univers": req.universe,
                "mode": req.mode,
                "liste_de_services": [req.serviceId],
                "date": datetime.now().isoformat(),
                "last_date": datetime.now().isoformat()
            }
        })

    # 2. Sauvegarde permanente dans MongoDB Atlas et sur disque
    save_to_mongo(req.msisdn, PROFILES_CACHE[req.msisdn])
    save_behavioral_click(req.msisdn, req.universe, req.mode, req.serviceId, req.reward)

    return {
        "status": "success",
        "msisdn": req.msisdn,
        "serviceId": req.serviceId,
        "reward": req.reward
    }

# Endpoint de l'API
@app.get("/api/v1/usages/stats/global")
def get_global_stats():
    service_counts = {}
    revenue = {
        "internet": 0,
        "illimix": 0,
        "illiflex": 0,
        "credit": 0
    }
    hourly_distribution = {
        "nuit": 0,
        "matin": 0,
        "aprem": 0,
        "soir": 0
    }
    
    # Pricing configuration matching the frontend
    pricing = {
        'TELCO.SERVICES.PASS.DATA': ('internet', 1000),
        'TELCO.SERVICES.PASS.ILLIMIX': ('illimix', 1500),
        'TELCO.SERVICES.PASS.ILLIFLEX': ('illiflex', 2000),
        'TELCO.SERVICES.PASS.VOICE': ('credit', 1000)
    }

    for msisdn, profiles in PROFILES_CACHE.items():
        for profile in profiles:
            source = profile.get("_source", {})
            services = source.get("liste_de_services", [])
            for srv in services:
                service_counts[srv] = service_counts.get(srv, 0) + 1
                if srv in pricing:
                    cat, price = pricing[srv]
                    revenue[cat] += price

            # Hourly distribution calculation
            dt_str = source.get("last_date") or source.get("date")
            if dt_str:
                try:
                    hour_str = None
                    if "T" in dt_str:
                        hour_str = dt_str.split("T")[1][:2]
                    elif " " in dt_str:
                        hour_str = dt_str.split(" ")[1][:2]
                    if hour_str and hour_str.isdigit():
                        hour = int(hour_str)
                        if hour < 6:
                            hourly_distribution["nuit"] += 1
                        elif hour < 12:
                            hourly_distribution["matin"] += 1
                        elif hour < 18:
                            hourly_distribution["aprem"] += 1
                        else:
                            hourly_distribution["soir"] += 1
                except Exception:
                    pass

    return {
        "serviceCounts": service_counts,
        "revenue": revenue,
        "hourlyDistribution": hourly_distribution
    }

@app.get("/api/v1/usages/{msisdn}")
def get_client_usage(msisdn: str):      
    # 1. Vérification dans le cache mémoire
    profil = PROFILES_CACHE.get(msisdn)

    # 2. Si absent de la mémoire, vérification dans MongoDB Atlas
    if not profil:
        mongo_p = get_from_mongo(msisdn)
        if mongo_p:
            PROFILES_CACHE[msisdn] = mongo_p
            profil = mongo_p

    # 3. Si toujours absent, relecture HDFS non-destructive
    if not profil:
        load_profile()
        profil = PROFILES_CACHE.get(msisdn)

    if not profil:
        # === TPS-09 : GESTION DES ERREURS (EXCEPTION HTTP 404 POUR MSISDN INCONNU) ===
        raise HTTPException(status_code=404, detail=f"Client {msisdn} introuvable")
    
    # Déduplication propre pour l'affichage sans répétition de services
    cleaned_profil = []
    for bloc in profil:
        src = bloc.get("_source", {})
        services = src.get("liste_de_services", [])
        # Conserve l'ordre initial d'apparition tout en supprimant les doublons
        unique_services = list(dict.fromkeys(services))
        
        cleaned_bloc = {
            **bloc,
            "_source": {
                **src,
                "liste_de_services": unique_services
            }
        }
        cleaned_profil.append(cleaned_bloc)

    return cleaned_profil

@app.get("/api/v1/clicks/{msisdn}")
@app.get("/api/v1/tracking/events/{msisdn}")
def get_user_tracking_events(msisdn: str):
    """Retourne la liste des événements enregistrés (clics, consultations et transactions) pour un client"""
    events = []
    _, clicks_coll = get_mongo_collections()
    
    # 1. Lecture dans MongoDB Atlas (collection behavioral_clicks & tracking_events)
    if clicks_coll is not None:
        try:
            docs = list(clicks_coll.find({"msisdn": msisdn}))
            for d in docs:
                d.pop("_id", None)
                if "actionType" not in d:
                    r = d.get("reward", 0.2)
                    d["actionType"] = "TRANSACTION" if r >= 1.0 else "CLICK"
                events.append(d)
        except Exception as e:
            logger.error("Erreur de lecture MongoDB pour les événements de %s : %s", msisdn, e)

    # 2. Complément depuis le fichier local si besoin
    if os.path.exists(BEHAVIORAL_CLICKS_FILE):
        try:
            with open(BEHAVIORAL_CLICKS_FILE, "r", encoding="utf-8") as f:
                local_clicks = json.load(f)
                for c in local_clicks:
                    if c.get("msisdn") == msisdn and c not in events:
                        if "actionType" not in c:
                            r = c.get("reward", 0.2)
                            c["actionType"] = "TRANSACTION" if r >= 1.0 else "CLICK"
                        events.append(c)
        except Exception:
            pass

    if not events:
        raise HTTPException(status_code=404, detail=f"Aucun événement de tracking trouvé pour le client {msisdn}")

    return {
        "msisdn": msisdn,
        "totalEvents": len(events),
        "events": events
    }

@app.get("/")
def read_root():
    return {
        "status": "L'API est connectée à MongoDB Atlas & HDFS !",
        "mongo_active": MONGO_CLIENT is not None,
        "msisdn_cached": len(PROFILES_CACHE)
    }