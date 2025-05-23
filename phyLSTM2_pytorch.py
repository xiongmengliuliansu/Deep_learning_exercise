import torch
import torch.nn as nn
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from random import shuffle
import time
import os

class DeepPhyLSTM(nn.Module):
    def __init__(self, eta, eta_t, g, ag, ag_c, lift, Phi_t, save_path, device='cuda'):
        super(DeepPhyLSTM, self).__init__()
        
        # Convert input data to PyTorch tensors and move to device
        self.eta = torch.tensor(eta, dtype=torch.float32).to(device)  # Displacement
        self.eta_t = torch.tensor(eta_t, dtype=torch.float32).to(device)  # Velocity
        self.g = torch.tensor(g, dtype=torch.float32).to(device)  # Restoring force
        self.ag = torch.tensor(ag, dtype=torch.float32).to(device)  # Ground acceleration
        self.ag_c = torch.tensor(ag_c, dtype=torch.float32).to(device)  # Collocation acceleration
        self.lift = torch.tensor(lift, dtype=torch.float32).to(device)  # Lift term
        self.Phi_t = torch.tensor(Phi_t, dtype=torch.float32).to(device)  # Finite difference matrix
        self.save_path = save_path
        self.device = device
        self.dof = eta.shape[2]  # Degrees of freedom
        
        # Define LSTM models
        self.lstm_model = LSTMModel(self.dof).to(device)
        self.lstm_model_f = LSTMModelF(self.dof).to(device)
        
        # Optimizer (Adam for initial training)
        self.optimizer_adam = torch.optim.Adam(self.parameters(), lr=1e-3)
        
    def LSTMModel(self, x):
        return self.lstm_model(x)
    
    def LSTMModelF(self, x):
        return self.lstm_model_f(x)
    
    def net_structure(self, ag, Phi_t_batch):
        """Predict state variables and compute derivatives using finite difference."""
        output = self.lstm_model(ag)
        eta = output[:, :, 0:self.dof]
        eta_dot = output[:, :, self.dof:2*self.dof]
        g = output[:, :, 2*self.dof:]
        eta_t = torch.matmul(Phi_t_batch, eta)
        eta_tt = torch.matmul(Phi_t_batch, eta_dot)
        return eta, eta_t, eta_tt, eta_dot, g
    
    def net_f(self, ag, Phi_t_batch):
        """Model dynamics and enforce physics at collocation points."""
        eta, eta_t, eta_tt, eta_dot, g = self.net_structure(ag, Phi_t_batch)
        eta_dot1 = eta_dot[:, :, 0:1]
        f = self.lstm_model_f(torch.cat([eta, eta_dot1, g], dim=2))
        lift = eta_tt + f
        return eta_t, eta_dot, lift
    
    def compute_loss(self, eta_tf, eta_t_tf, g_tf, ag_tf, lift_tf, ag_c_tf, Phi_tf):
        """Compute total loss including data and physics constraints."""
        batch_size = ag_tf.shape[0]
        Phi_t_batch = self.Phi_t[:batch_size]
        eta_pred, eta_t_pred, eta_tt_pred, eta_dot_pred, g_pred = self.net_structure(ag_tf, Phi_t_batch)
        batch_size_c = ag_c_tf.shape[0]
        Phi_t_batch_c = self.Phi_t[:batch_size_c]
        eta_t_pred_c, eta_dot_pred_c, lift_c_pred = self.net_f(ag_c_tf, Phi_t_batch_c)
        loss_u = torch.mean(torch.pow(eta_tf - eta_pred, 2))
        loss_udot = torch.mean(torch.pow(eta_t_tf - eta_dot_pred, 2))
        loss_g = torch.mean(torch.pow(g_tf - g_pred, 2))
        loss_ut_c = torch.mean(torch.pow(eta_t_pred_c - eta_dot_pred_c, 2))
        ones = torch.ones([lift_tf.shape[0], 1, self.dof], dtype=torch.float32).to(self.device)
        loss_e = torch.mean(torch.pow(torch.matmul(lift_tf, ones) - lift_c_pred, 2))
        total_loss = loss_u + loss_udot + loss_g + loss_ut_c + loss_e
        return total_loss, loss_u, loss_udot, loss_g, loss_ut_c, loss_e
    
    def train(self, num_epochs, learning_rate, bfgs=False):
        Loss_u, Loss_udot, Loss_g, Loss_ut_c, Loss_e = [], [], [], [], []
        Loss, Loss_val = [], []
        best_loss = 100.0
        
        for epoch in range(num_epochs):
            Ind = list(range(self.ag.shape[0]))
            shuffle(Ind)
            ratio_split = 0.8
            Ind_tr = Ind[:int(ratio_split * self.ag.shape[0])]
            Ind_val = Ind[int(ratio_split * self.ag.shape[0]):]
            
            ag_tr = self.ag[Ind_tr]
            eta_tr = self.eta[Ind_tr]
            eta_t_tr = self.eta_t[Ind_tr]
            g_tr = self.g[Ind_tr]
            ag_val = self.ag[Ind_val]
            eta_val = self.eta[Ind_val]
            eta_t_val = self.eta_t[Ind_val]
            g_val = self.g[Ind_val]
            
            start_time = time.time()
            self.optimizer_adam.zero_grad()
            loss, loss_u, loss_udot, loss_g, loss_ut_c, loss_e = self.compute_loss(
                eta_tr, eta_t_tr, g_tr, ag_tr, self.lift, self.ag_c, self.Phi_t
            )
            loss.backward()
            self.optimizer_adam.step()
            
            with torch.no_grad():
                loss_val, _, _, _, _, _ = self.compute_loss(
                    eta_val, eta_t_val, g_val, ag_val, self.lift, self.ag_c, self.Phi_t
                )
            
            Loss_u.append(loss_u.item())
            Loss_udot.append(loss_udot.item())
            Loss_g.append(loss_g.item())
            Loss_ut_c.append(loss_ut_c.item())
            Loss_e.append(loss_e.item())
            Loss.append(loss.item())
            Loss_val.append(loss_val.item())
            
            elapsed = time.time() - start_time
            print(f'Epoch: {epoch}, Loss: {loss.item():.3e}, Loss_val: {loss_val.item():.3e}, '
                  f'Best_loss: {best_loss:.3e}, Time: {elapsed:.2f}, Learning Rate: {learning_rate:.3e}')
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(self.state_dict(), self.save_path)
        
        if bfgs:
            optimizer_lbfgs = torch.optim.LBFGS(self.parameters(), max_iter=20000, max_eval=50000, tolerance_grad=1e-7)
            def closure():
                optimizer_lbfgs.zero_grad()
                loss, _, _, _, _, _ = self.compute_loss(
                    eta_tr, eta_t_tr, g_tr, ag_tr, self.lift, self.ag_c, self.Phi_t
                )
                loss.backward()
                return loss
            optimizer_lbfgs.step(closure)
            with torch.no_grad():
                loss, loss_u, loss_udot, loss_g, loss_ut_c, loss_e = self.compute_loss(
                    eta_tr, eta_t_tr, g_tr, ag_tr, self.lift, self.ag_c, self.Phi_t
                )
                loss_val, _, _, _, _, _ = self.compute_loss(
                    eta_val, eta_t_val, g_val, ag_val, self.lift, self.ag_c, self.Phi_t
                )
            
            Loss_u.append(loss_u.item())
            Loss_udot.append(loss_udot.item())
            Loss_g.append(loss_g.item())
            Loss_ut_c.append(loss_ut_c.item())
            Loss_e.append(loss_e.item())
            Loss.append(loss.item())
            Loss_val.append(loss_val.item())
        
        return Loss_u, Loss_udot, Loss_g, Loss_ut_c, Loss_e, Loss, Loss_val, best_loss
    
    def predict(self, ag_star, Phi_t0):
        """Predict state variables for given ground acceleration."""
        ag_star = torch.tensor(ag_star, dtype=torch.float32).to(self.device)
        batch_size = ag_star.shape[0]
        Phi_t_batch = np.repeat(Phi_t0, batch_size, axis=0)
        Phi_t_batch = torch.tensor(Phi_t_batch, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            eta, eta_t, eta_tt, eta_dot, g = self.net_structure(ag_star, Phi_t_batch)
        return eta.cpu().numpy(), eta_t.cpu().numpy(), eta_tt.cpu().numpy(), eta_dot.cpu().numpy(), g.cpu().numpy()
    
    def predict_c(self, ag_star, Phi_t0):
        """Predict lift term at collocation points."""
        ag_star = torch.tensor(ag_star, dtype=torch.float32).to(self.device)
        batch_size = ag_star.shape[0]
        Phi_t_batch = np.repeat(Phi_t0, batch_size, axis=0)
        Phi_t_batch = torch.tensor(Phi_t_batch, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            eta_t, eta_dot, lift = self.net_f(ag_star, Phi_t_batch)
        return lift.cpu().numpy()
    
    def predict_best_model(self, path, ag_star, Phi_t0):
        """Load best model and predict."""
        self.load_state_dict(torch.load(path))
        return self.predict(ag_star, Phi_t0)

class LSTMModel(nn.Module):
    def __init__(self, dof):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=100, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=100, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=100, hidden_size=100, num_layers=1, batch_first=True)
        self.dense1 = nn.Linear(100, 100)
        self.dense2 = nn.Linear(100, 3 * dof)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.relu(x)
        x, _ = self.lstm2(x)
        x = self.relu(x)
        x, _ = self.lstm3(x)
        x = self.relu(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

class LSTMModelF(nn.Module):
    def __init__(self, dof):
        super(LSTMModelF, self).__init__()
        self.lstm1 = nn.LSTM(input_size=3*dof, hidden_size=100, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=100, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=100, hidden_size=100, num_layers=1, batch_first=True)
        self.dense1 = nn.Linear(100, 100)
        self.dense2 = nn.Linear(100, dof)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.relu(x)
        x, _ = self.lstm2(x)
        x = self.relu(x)
        x, _ = self.lstm3(x)
        x = self.relu(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

if __name__ == "__main__":
    # Load data
    dataDir = "/share/home/neallee/hkf/phyLSTM/data/"
    mat = scipy.io.loadmat(dataDir + 'data_boucwen.mat')
    
    ag_data = mat['input_tf']
    u_data = mat['target_X_tf']
    ut_data = mat['target_Xd_tf']
    utt_data = mat['target_Xdd_tf']
    ag_data = ag_data.reshape([ag_data.shape[0], ag_data.shape[1], 1])
    u_data = u_data.reshape([u_data.shape[0], u_data.shape[1], 1])
    ut_data = ut_data.reshape([ut_data.shape[0], ut_data.shape[1], 1])
    utt_data = utt_data.reshape([utt_data.shape[0], utt_data.shape[1], 1])
    
    t = mat['time']
    dt = t[0, 1] - t[0, 0]
    time_array = t[0, :]
    
    ag_all = ag_data
    u_all = u_data
    u_t_all = ut_data
    u_tt_all = utt_data
    
    # Finite difference matrix
    n = u_data.shape[1]
    phi1 = np.concatenate([np.array([-3 / 2, 2, -1 / 2]), np.zeros([n - 3, ])])
    temp1 = np.concatenate([-1 / 2 * np.identity(n - 2), np.zeros([n - 2, 2])], axis=1)
    temp2 = np.concatenate([np.zeros([n - 2, 2]), 1 / 2 * np.identity(n - 2)], axis=1)
    phi2 = temp1 + temp2
    phi3 = np.concatenate([np.zeros([n - 3, ]), np.array([1 / 2, -2, 3 / 2])])
    Phi_t0 = 1 / dt * np.concatenate(
        [np.reshape(phi1, [1, phi1.shape[0]]), phi2, np.reshape(phi3, [1, phi3.shape[0]])], axis=0)
    Phi_t0 = np.reshape(Phi_t0, [1, n, n])
    
    ag_star = ag_all[0:10]
    eta_star = u_all[0:10]
    eta_t_star = u_t_all[0:10]
    eta_tt_star = u_tt_all[0:10]
    ag_c_star = ag_all[0:50]
    lift_star = -ag_c_star
    g = -eta_tt_star - ag_star
    
    eta = eta_star
    ag = ag_star
    lift = lift_star
    eta_t = eta_t_star
    eta_tt = eta_tt_star
    ag_c = ag_c_star
    g = g
    Phi_t = np.repeat(Phi_t0, ag_c_star.shape[0], axis=0)
    
    # Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = 'model.pth'
    model = DeepPhyLSTM(eta, eta_t, g, ag, ag_c, lift, Phi_t, save_path=save_path, device=device)
    
    Loss_u, Loss_udot, Loss_g, Loss_ut_c, Loss_e, Loss, Loss_val, best_loss = model.train(
        num_epochs=5000, learning_rate=1e-3, bfgs=False
    )
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(dataDir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
        # Plot loss
    plt.figure()
    plt.plot(np.log(Loss), label='Training Loss')
    plt.plot(np.log(Loss_val), label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'loss_plot.png'))
    plt.close()
    
    # Training performance
    X_train = ag_data[0:10]
    y_train_ref = u_data[0:10]
    yt_train_ref = ut_data[0:10]
    ytt_train_ref = utt_data[0:10]
    lift_train_ref = -X_train
    g_train_ref = -ytt_train_ref + lift_train_ref
    
    eta, eta_t, eta_tt, eta_dot, g = model.predict(X_train, Phi_t0)
    lift = model.predict_c(X_train, Phi_t0)
    y_train_pred = eta
    yt_train_pred = eta_t
    ytt_train_pred = eta_tt
    g_train_pred = -eta_tt + lift
    
    # Compute correlation coefficients for training
    gamma_train_u = np.zeros(len(y_train_ref))
    gamma_train_ut = np.zeros(len(y_train_ref))
    gamma_train_utt = np.zeros(len(y_train_ref))
    gamma_train_g = np.zeros(len(y_train_ref))
    
    dof = 0
    for ii in range(len(y_train_ref)):
        # Compute correlation coefficients
        gamma_train_u[ii] = np.corrcoef(y_train_ref[ii, :, dof].flatten(), y_train_pred[ii, :, dof].flatten())[0, 1]
        gamma_train_ut[ii] = np.corrcoef(yt_train_ref[ii, :, dof].flatten(), yt_train_pred[ii, :, dof].flatten())[0, 1]
        gamma_train_utt[ii] = np.corrcoef(ytt_train_ref[ii, :, dof].flatten(), ytt_train_pred[ii, :, dof].flatten())[0, 1]
        gamma_train_g[ii] = np.corrcoef(g_train_ref[ii, :, dof].flatten(), g_train_pred[ii, :, dof].flatten())[0, 1]
        
        # Plot displacement
        plt.figure()
        plt.plot(time_array, y_train_ref[ii, :, dof], '-', label='True')
        plt.plot(time_array, y_train_pred[ii, :, dof], '--', label=f'Predicted (γ={gamma_train_u[ii]:.2f})')
        plt.title('Training Displacement')
        plt.xlabel('Time (s)')
        plt.ylabel('Displacement (m)')
        plt.xlim(0, 30)
        plt.legend()
        plt.savefig(os.path.join(results_dir, f'training_u_{ii}.png'))
        plt.close()
    
        # Plot velocity
        plt.figure()
        plt.plot(time_array, yt_train_ref[ii, :, dof], '-', label='True')
        plt.plot(time_array, yt_train_pred[ii, :, dof], '--', label=f'Predicted (γ={gamma_train_ut[ii]:.2f})')
        plt.title('Training Velocity')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.xlim(0, 30)
        plt.legend()
        plt.savefig(os.path.join(results_dir, f'training_u_t_{ii}.png'))
        plt.close()
    
        # Plot acceleration
        plt.figure()
        plt.plot(time_array, ytt_train_ref[ii, :, dof], '-', label='True')
        plt.plot(time_array, ytt_train_pred[ii, :, dof], '--', label=f'Predicted (γ={gamma_train_utt[ii]:.2f})')
        plt.title('Training Acceleration')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s²)')
        plt.xlim(0, 30)
        plt.legend()
        plt.savefig(os.path.join(results_dir, f'training_u_tt_{ii}.png'))
        plt.close()
    
        # Plot restoring force
        plt.figure()
        plt.plot(time_array, g_train_ref[ii, :, dof], '-', label='True')
        plt.plot(time_array, g_train_pred[ii, :, dof], '--', label=f'Predicted (γ={gamma_train_g[ii]:.2f})')
        plt.title('Training Restoring Force')
        plt.xlabel('Time (s)')
        plt.ylabel('Restoring Force (N/kg)')
        plt.xlim(0, 30)
        plt.legend()
        plt.savefig(os.path.join(results_dir, f'training_g_{ii}.png'))
        plt.close()
    
        # Plot hysteresis
        plt.figure()
        plt.plot(y_train_ref[ii, :, dof], g_train_ref[ii, :, dof], '-', label='True')
        plt.plot(y_train_pred[ii, :, dof], g_train_pred[ii, :, dof], '--', label='Predicted')
        plt.title('Training Hysteresis')
        plt.xlabel('Displacement (m)')
        plt.ylabel('Restoring Force (N/kg)')
        plt.legend()
        plt.savefig(os.path.join(results_dir, f'training_hysteresis_{ii}.png'))
        plt.close()
    
    # Prediction performance
    X_pred = ag_data[10:]
    y_pred_ref = u_data[10:]
    yt_pred_ref = ut_data[10:]
    ytt_pred_ref = utt_data[10:]
    lift_pred_ref = -X_pred
    g_pred_ref = -ytt_pred_ref + lift_pred_ref
    
    eta, eta_t, eta_tt, eta_dot, g = model.predict(X_pred, Phi_t0)
    lift = model.predict_c(X_pred, Phi_t0)
    y_pred = eta
    yt_pred = eta_t
    ytt_pred = eta_tt
    g_pred = -eta_tt + lift
    
    # Compute correlation coefficients for prediction
    gamma_pred_u = np.zeros(len(y_pred_ref))
    gamma_pred_ut = np.zeros(len(y_pred_ref))
    gamma_pred_utt = np.zeros(len(y_pred_ref))
    gamma_pred_g = np.zeros(len(y_pred_ref))
    
    for ii in range(len(y_pred_ref)):
        # Compute correlation coefficients
        gamma_pred_u[ii] = np.corrcoef(y_pred_ref[ii, :, dof].flatten(), y_pred[ii, :, dof].flatten())[0, 1]
        gamma_pred_ut[ii] = np.corrcoef(yt_pred_ref[ii, :, dof].flatten(), yt_pred[ii, :, dof].flatten())[0, 1]
        gamma_pred_utt[ii] = np.corrcoef(ytt_pred_ref[ii, :, dof].flatten(), ytt_pred[ii, :, dof].flatten())[0, 1]
        gamma_pred_g[ii] = np.corrcoef(g_pred_ref[ii, :, dof].flatten(), g_pred[ii, :, dof].flatten())[0, 1]
        
        # Plot displacement
        plt.figure()
        plt.plot(time_array, y_pred_ref[ii, :, dof], '-', label='True')
        plt.plot(time_array, y_pred[ii, :, dof], '--', label=f'Predicted (γ={gamma_pred_u[ii]:.2f})')
        plt.title('Prediction Displacement')
        plt.xlabel('Time (s)')
        plt.ylabel('Displacement (m)')
        plt.xlim(0, 30)
        plt.legend()
        plt.savefig(os.path.join(results_dir, f'prediction_u_{ii}.png'))
        plt.close()
    
        # Plot velocity
        plt.figure()
        plt.plot(time_array, yt_pred_ref[ii, :, dof], '-', label='True')
        plt.plot(time_array, yt_pred[ii, :, dof], '--', label=f'Predicted (γ={gamma_pred_ut[ii]:.2f})')
        plt.title('Prediction Velocity')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.xlim(0, 30)
        plt.legend()
        plt.savefig(os.path.join(results_dir, f'prediction_u_t_{ii}.png'))
        plt.close()
    
        # Plot acceleration
        plt.figure()
        plt.plot(time_array, ytt_pred_ref[ii, :, dof], '-', label='True')
        plt.plot(time_array, ytt_pred[ii, :, dof], '--', label=f'Predicted (γ={gamma_pred_utt[ii]:.2f})')
        plt.title('Prediction Acceleration')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s²)')
        plt.xlim(0, 30)
        plt.legend()
        plt.savefig(os.path.join(results_dir, f'prediction_u_tt_{ii}.png'))
        plt.close()
    
        # Plot restoring force
        plt.figure()
        plt.plot(time_array, g_pred_ref[ii, :, dof], '-', label='True')
        plt.plot(time_array, g_pred[ii, :, dof], '--', label=f'Predicted (γ={gamma_pred_g[ii]:.2f})')
        plt.title('Prediction Restoring Force')
        plt.xlabel('Time (s)')
        plt.ylabel('Restoring Force (N/kg)')
        plt.xlim(0, 30)
        plt.legend()
        plt.savefig(os.path.join(results_dir, f'prediction_g_{ii}.png'))
        plt.close()
    
        # Plot hysteresis
        plt.figure()
        plt.plot(y_pred_ref[ii, :, dof], g_pred_ref[ii, :, dof], '-', label='True')
        plt.plot(y_pred[ii, :, dof], g_pred[ii, :, dof], '--', label='Predicted')
        plt.title('Prediction Hysteresis')
        plt.xlabel('Displacement (m)')
        plt.ylabel('Restoring Force (N/kg)')
        plt.legend()
        plt.savefig(os.path.join(results_dir, f'prediction_hysteresis_{ii}.png'))
        plt.close()
    
    # Plot histograms of correlation coefficients for training
    # Plot histograms of correlation coefficients for training with Probability (%)
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.hist(gamma_train_u, bins=20, alpha=0.7, label='Displacement', weights=np.ones_like(gamma_train_u)/len(gamma_train_u)*100)
    plt.xlabel('Correlation Coefficient (γ)')
    plt.ylabel('Probability (%)')
    plt.title('Training Displacement γ')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.hist(gamma_train_ut, bins=20, alpha=0.7, label='Velocity', weights=np.ones_like(gamma_train_ut)/len(gamma_train_ut)*100)
    plt.xlabel('Correlation Coefficient (γ)')
    plt.ylabel('Probability (%)')
    plt.title('Training Velocity γ')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.hist(gamma_train_utt, bins=20, alpha=0.7, label='Acceleration', weights=np.ones_like(gamma_train_utt)/len(gamma_train_utt)*100)
    plt.xlabel('Correlation Coefficient (γ)')
    plt.ylabel('Probability (%)')
    plt.title('Training Acceleration γ')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.hist(gamma_train_g, bins=20, alpha=0.7, label='Restoring Force', weights=np.ones_like(gamma_train_g)/len(gamma_train_g)*100)
    plt.xlabel('Correlation Coefficient (γ)')
    plt.ylabel('Probability (%)')
    plt.title('Training Restoring Force γ')
    plt.legend()

    plt.tight_layout()
    training_hist_path = os.path.join(results_dir, 'training_gamma_hist.png')
    plt.savefig(training_hist_path)
    plt.close()
    print(f"Saved training histogram to {training_hist_path}")

    # Plot histograms of correlation coefficients for prediction with Probability (%)
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.hist(gamma_pred_u, bins=20, alpha=0.7, label='Displacement', weights=np.ones_like(gamma_pred_u)/len(gamma_pred_u)*100)
    plt.xlabel('Correlation Coefficient (γ)')
    plt.ylabel('Probability (%)')
    plt.title('Prediction Displacement γ')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.hist(gamma_pred_ut, bins=20, alpha=0.7, label='Velocity', weights=np.ones_like(gamma_pred_ut)/len(gamma_pred_ut)*100)
    plt.xlabel('Correlation Coefficient (γ)')
    plt.ylabel('Probability (%)')
    plt.title('Prediction Velocity γ')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.hist(gamma_pred_utt, bins=20, alpha=0.7, label='Acceleration', weights=np.ones_like(gamma_pred_utt)/len(gamma_pred_utt)*100)
    plt.xlabel('Correlation Coefficient (γ)')
    plt.ylabel('Probability (%)')
    plt.title('Prediction Acceleration γ')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.hist(gamma_pred_g, bins=20, alpha=0.7, label='Restoring Force', weights=np.ones_like(gamma_pred_g)/len(gamma_pred_g)*100)
    plt.xlabel('Correlation Coefficient (γ)')
    plt.ylabel('Probability (%)')
    plt.title('Prediction Restoring Force γ')
    plt.legend()

    plt.tight_layout()
    prediction_hist_path = os.path.join(results_dir, 'prediction_gamma_hist.png')
    plt.savefig(prediction_hist_path)
    plt.close()
    print(f"Saved prediction histogram to {prediction_hist_path}")
    
    # Save results
    scipy.io.savemat(os.path.join(results_dir, 'results_PhyLSTM2_pytorch.mat'),
                     {'y_train_ref': y_train_ref, 'yt_train_ref': yt_train_ref, 'ytt_train_ref': ytt_train_ref,
                      'g_train_ref': g_train_ref, 'y_train_pred': y_train_pred, 'yt_train_pred': yt_train_pred,
                      'ytt_train_pred': ytt_train_pred, 'g_train_pred': g_train_pred, 'y_pred_ref': y_pred_ref,
                      'yt_pred_ref': yt_pred_ref, 'ytt_pred_ref': ytt_pred_ref, 'g_pred_ref': g_pred_ref,
                      'y_pred': y_pred, 'yt_pred': yt_pred, 'ytt_pred': ytt_pred, 'g_pred': g_pred,
                      'X_train': X_train, 'X_pred': X_pred, 'time': t, 'dt': dt,
                      'train_loss': Loss, 'test_loss': Loss_val, 'best_loss': best_loss,
                      'gamma_train_u': gamma_train_u, 'gamma_train_ut': gamma_train_ut,
                      'gamma_train_utt': gamma_train_utt, 'gamma_train_g': gamma_train_g,
                      'gamma_pred_u': gamma_pred_u, 'gamma_pred_ut': gamma_pred_ut,
                      'gamma_pred_utt': gamma_pred_utt, 'gamma_pred_g': gamma_pred_g})
    
    # Print summary of correlation coefficients
    print(f"Training γ_u: Min={gamma_train_u.min():.2f}, Max={gamma_train_u.max():.2f}, Mean={gamma_train_u.mean():.2f}")
    print(f"Prediction γ_u: Min={gamma_pred_u.min():.2f}, Max={gamma_pred_u.max():.2f}, Mean={gamma_pred_u.mean():.2f}")