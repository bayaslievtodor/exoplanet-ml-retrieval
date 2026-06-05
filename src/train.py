import numpy as np
from sklearn.metrics import r2_score
import torch
import torch.nn as nn

from src.models import build_random_forest
from src.utils import pinball_loss


# RF train
def train_random_forest(X_train_scaled, y_train, X_val_scaled, y_val,
                        n_estimators=100, max_depth=15,
                        random_state=123, n_jobs=-1):

    #also returns the r2
    rf = build_random_forest(n_estimators=n_estimators,
                             max_depth=max_depth,
                             random_state=random_state,
                             n_jobs=n_jobs)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_val_scaled)
    r2_scores = r2_score(y_val, y_pred, multioutput='raw_values')
    return rf, y_pred, r2_scores


# CNN train
def train_cnn(model, train_loader, val_loader,
              epochs=100, lr=1e-3, device='cpu',
              print_every=10):
              
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    
    loss_fn = pinball_loss

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        model.train()
        batch_losses = []
        for spec_b, aux_b, y_b in train_loader:
            spec_b, aux_b, y_b = spec_b.to(device), aux_b.to(device), y_b.to(device)
            optimiser.zero_grad()
            pred = model(spec_b, aux_b)
            
            loss = loss_fn(pred, y_b)
            
            loss.backward()
            optimiser.step()
            batch_losses.append(loss.item())
        train_loss = np.mean(batch_losses)

        model.eval()
        with torch.no_grad():
            val_batch_losses = []
            for spec_b, aux_b, y_b in val_loader:
                spec_b, aux_b, y_b = spec_b.to(device), aux_b.to(device), y_b.to(device)
                pred = model(spec_b, aux_b)
                
                val_batch_losses.append(loss_fn(pred, y_b).item())
            val_loss = np.mean(val_batch_losses)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} — "
                  f"train loss: {train_loss:.4f} — "
                  f"val loss: {val_loss:.4f}")

    model.eval()
    y_pred_list = []
    y_val_list  = []
    with torch.no_grad():
        for spec_b, aux_b, y_b in val_loader:
            spec_b, aux_b = spec_b.to(device), aux_b.to(device)
            pred = model(spec_b, aux_b)
            y_pred_list.append(pred.cpu())
            y_val_list.append(y_b)
    y_pred = torch.cat(y_pred_list, dim=0).numpy()
    y_val  = torch.cat(y_val_list,  dim=0).numpy()

    r2_cnn = r2_score(y_val, y_pred, multioutput='raw_values')
    return model, train_losses, val_losses, y_pred, y_val, r2_cnn