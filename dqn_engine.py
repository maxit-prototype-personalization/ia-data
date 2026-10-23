import random
import math 
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# Catalogue complet des services Max IT (Espace d'Actions)
ALL_SERVICES = [
    # TELCO Services
    "TELCO.SERVICES.PASS.VOICE",
    "TELCO.SERVICES.PASS.ILLIFLEX",
    "TELCO.SERVICES.PASS.ILLIMIX",
    "TELCO.SERVICES.PASS.DATA",
    "TELCO.SERVICES.PASS.INTERNATIONAL",
    "TELCO.SERVICES.HOME_INTERNET",
    "TELCO.SERVICES.LOYALTY",
    "TELCO.SERVICES.SOS_CREDIT",
    "TELCO.SERVICES.P2P_BONUS",
    "TELCO.SERVICES.PASS.MIXEL",
    "SONATELF_MAIN_PAGE",
    "TELCO.SERVICES.PASS.TRAVEL",
    "TELCO.SERVICES.P2P",
    "TELCO.SERVICES.PASS.WIDO",
    "TELCO.SERVICES.LEISURE.DALAL",
    "TELCO.SERVICES.SOS_PASS",
    # OMY Services
    "OMY.SERVICES.VOICEBUNDLE",
    "RAPIDO",
    "OMY.SERVICES.TRANSFERT",
    "DEPOT",
    "OMY.SERVICES.RETRAIT",
    "OMY.SERVICES.MASTERCARD",
    "OMY.SERVICES.FINANCIERS",
    "OMY.SERVICES.PI",
    "OMY.SERVICES.INTER_COMPTE",
    "OMY.SERVICES.BILLERS",
    "OMY.SERVICES.BILLERS.TER",
    "OBA.SERVICES.TIKTAK",
    "CANALPLUS",
    "FLEXEAU",
    "SENELEC_MAIN_PAGE",
]

SERVICE_TO_INDEX = {srv: i for i, srv in enumerate(ALL_SERVICES)}
INDEX_TO_SERVICE = {i: srv for i, srv in enumerate(ALL_SERVICES)}

STATE_DIM = 8
ACTION_DIM = len(ALL_SERVICES)

# 1. Définition du Réseau de Neurones Profond (PyTorch Q-Network)
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(64, action_dim)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.out(x)

# 2. Memory Buffer pour l'Experience Replay
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.FloatTensor(np.array(state)),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(done),
        )

    def __len__(self):
        return len(self.buffer)

# 3. Agent Deep Q-Network Recommender
class DQNRecommender:
    def __init__(self, gamma=0.95, epsilon=0.2, lr=0.001, batch_size=32):
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        
        self.policy_net = QNetwork(STATE_DIM, ACTION_DIM)
        self.target_net = QNetwork(STATE_DIM, ACTION_DIM)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = ReplayBuffer(capacity=10000)
        
        self.train_step_count = 0
        self.update_target_every = 50

    def build_state_vector(self, telco_count=0, omy_count=0, hour=12, mode="SIMPLE", universe="TELCO"):
        # === TDQN-01 : CONSTRUCTION DU VECTEUR D'ÉTAT S (DIMENSION = 8) ===
        tot = max(1, telco_count + omy_count)
        telco_ratio = telco_count / tot
        omy_ratio = omy_count / tot
        
        # Encodage cyclique de l'heure
        rad = 2 * math.pi * (hour % 24) / 24.0
        sin_h = math.sin(rad)
        cos_h = math.cos(rad)
        
        is_simple = 1.0 if mode == "SIMPLE" else 0.0
        is_advance = 1.0 if mode == "ADVANCE" else 0.0
        is_telco = 1.0 if universe == "TELCO" else 0.0
        is_omy = 1.0 if universe == "OMY" else 0.0
        
        return [telco_ratio, omy_ratio, sin_h, cos_h, is_simple, is_advance, is_telco, is_omy]

    def warm_start_from_hdfs(self, profiles_cache):
        """Pré-entraînement / Warm-Start sur l'historique HDFS"""
        if self.load_model():
            print("Modèle PyTorch réhydraté depuis dqn_model.pth")
            return

        print("Début du Warm-Start sur l'historique HDFS...")
        pushed_count = 0
        
        for msisdn, profiles in profiles_cache.items():
            telco_c = 0
            omy_c = 0
            # Compter les occurrences par univers
            for profile in profiles:
                src = profile.get("_source", {})
                if src.get("univers") == "TELCO":
                    telco_c += len(src.get("liste_de_services", []))
                elif src.get("univers") == "OMY":
                    omy_c += len(src.get("liste_de_services", []))
            
            for profile in profiles:
                src = profile.get("_source", {})
                univers = src.get("univers", "TELCO")
                mode = src.get("mode", "SIMPLE")
                services = src.get("liste_de_services", [])
                
                state = self.build_state_vector(
                    telco_count=telco_c, 
                    omy_count=omy_c, 
                    hour=14, 
                    mode=mode, 
                    universe=univers
                )
                
                for srv in services:
                    if srv in SERVICE_TO_INDEX:
                        action_idx = SERVICE_TO_INDEX[srv]
                        reward = 1.0  # Récompense positive pour un usage historique avéré
                        next_state = state  # État suivant identique pour le bootstrap
                        self.memory.push(state, action_idx, reward, next_state, False)
                        pushed_count += 1

        print(f"ReplayBuffer alimenté avec {pushed_count} transitions historiques.")
        
        # Entraîner le réseau sur plusieurs itérations pour fixer les poids initiaux
        if len(self.memory) >= self.batch_size:
            for _ in range(20):
                self.optimize_model()
            self.save_model()
            print("Modèle PyTorch DQN pré-entraîné et sauvegardé avec succès !")

    def predict_recommendations(self, state_vector, univers="TELCO", top_n=10, user_counts=None):
        """Retourne la liste des services ordonnés par les Q-values calculées par le réseau PyTorch + Score Hybride Personnalisé"""
        self.policy_net.eval()
        with torch.no_grad():
            # === TDQN-02 : TRANSMISSION DU VECTEUR D'ÉTAT AU MODÈLE (TENSOR PYTORCH) ===
            state_t = torch.FloatTensor(state_vector).unsqueeze(0)
            
            # === TDQN-03 : INFÉRENCE DU MODÈLE QNETWORK (CALCUL DES Q-VALUES) ===
            q_values = self.policy_net(state_t).squeeze(0).numpy()

        # === TDQN-04 : SÉLECTION DES ACTIONS ET FILTRAGE PAR UNIVERS (TELCO / OMY) ===
        scored_services = []
        for srv, idx in SERVICE_TO_INDEX.items():
            # Vérifier l'appartenance à l'univers
            if univers == "TELCO" and not srv.startswith("TELCO") and not srv.startswith("SONATELF"):
                continue
            if univers == "OMY" and not srv.startswith("OMY") and srv not in ["RAPIDO", "DEPOT", "OBA.SERVICES.TIKTAK", "CANALPLUS", "FLEXEAU", "SENELEC_MAIN_PAGE"]:
                continue
                
            q_val = float(q_values[idx])

            # === TDQN-05 : PRISE EN COMPTE DE LA FRÉQUENCE D'UTILISATION (SCORE HYBRIDE) ===
            if user_counts and srv in user_counts:
                count = user_counts[srv]
                bonus = math.log1p(count) * 0.2
                q_val += bonus

            scored_services.append((srv, q_val))

        # === TDQN-06 : GÉNÉRATION DU CLASSEMENT (TRI DÉCROISSANT PAR SCORE HYBRIDE) ===
        scored_services.sort(key=lambda x: x[1], reverse=True)
        
        # === TDQN-08 : GÉNÉRATION DU TOP N RECOMMANDATIONS ===
        return scored_services[:top_n]

    def record_feedback(self, state_vector, service_id, reward):
        """Enregistre une transition de récompense, optimise immédiatement le réseau et persiste les poids sur disque"""
        # === TDQN-07 : PRISE EN COMPTE DU FEEDBACK (REPLAY BUFFER & RÉTROPROPAGATION) ===
        if service_id not in SERVICE_TO_INDEX:
            return
            
        action_idx = SERVICE_TO_INDEX[service_id]
        next_state = state_vector
        self.memory.push(state_vector, action_idx, reward, next_state, False)
        
        # Optimisation immédiate du modèle dès le 1er clic
        self.optimize_model()
        
        # Sauvegarde continue des poids mis à jour
        self.save_model()

    def optimize_model(self):
        if len(self.memory) == 0:
            return
            
        cur_batch = min(len(self.memory), self.batch_size)
        self.policy_net.train()
        states, actions, rewards, next_states, dones = self.memory.sample(cur_batch)

        # Q(s, a) courant
        state_action_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # max_a Q_target(s', a)
        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            expected_state_action_values = rewards + (self.gamma * next_state_values * (1 - dones))

        # Loss MSE entre Q prédit et Q cible
        loss = self.criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.save_model()

    def save_model(self, filepath="dqn_model.pth"):
        """Sauvegarde les poids du réseau de neurones sur disque"""
        try:
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, filepath)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du modèle : {e}")

    def load_model(self, filepath="dqn_model.pth"):
        """Charge les poids enregistrés du modèle depuis le disque s'il existe"""
        import os
        if os.path.exists(filepath):
            try:
                checkpoint = torch.load(filepath)
                self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Modèle PyTorch réhydraté avec succès depuis '{filepath}'.")
                return True
            except Exception as e:
                print(f"Échec du chargement du fichier modèle : {e}")
        return False

# Instance singleton du moteur DQN
dqn_recommender = DQNRecommender()
