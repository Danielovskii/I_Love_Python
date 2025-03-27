import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('creditCardDefault_train.csv')
test_data = pd.read_csv('creditCardDefault_test.csv')

predictor_names = ['creditLimit', 'gender', 'edu', 'age', 'nDelay', 'billAmt1', 'billAmt2', 
                  'billAmt3', 'billAmt4', 'billAmt5', 'billAmt6']
X_train = train_data[predictor_names]
y_train = train_data['default'].astype(int)
X_test = test_data[predictor_names]
y_test = test_data['default'].astype(int)

# Escalar los datos para Regresión Logística
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Regresión Logística
log_reg = LogisticRegression(max_iter=2000)
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)
y_prob_log = log_reg.predict_proba(X_test_scaled)[:, 1]
log_accuracy = accuracy_score(y_test, y_pred_log)
log_cm = confusion_matrix(y_test, y_pred_log)

#Random Forest
accuracy_train = []
accuracy_test = []
depths = range(1, 26)
for i in depths:
    rf = RandomForestClassifier(max_depth=i, random_state=16)
    rf.fit(X_train, y_train)  # Random Forest no necesita escalado
    accuracy_test.append(accuracy_score(y_test, rf.predict(X_test)))
    accuracy_train.append(accuracy_score(y_train, rf.predict(X_train)))

# Encontrar el mejor max_depth
best_acc = np.max(accuracy_test)
best_depth = depths[np.argmax(accuracy_test)]
print(f"Mejor max_depth para Random Forest: {best_depth}")
print(f"Mejor precisión en el conjunto de prueba: {round(best_acc*100, 3)}%")

plt.figure(figsize=(8, 6))
plt.plot(depths, accuracy_test, 'bo--', label='Test Accuracy')
plt.plot(depths, accuracy_train, 'r*:', label='Train Accuracy')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Random Forest Accuracy vs Max Depth')
plt.legend()
plt.grid(True)
plt.show()

# Entrenar el mejor modelo Random Forest
best_rf = RandomForestClassifier(max_depth=best_depth, random_state=16)
best_rf.fit(X_train, y_train)
y_pred_rf = best_rf.predict(X_test)
y_prob_rf = best_rf.predict_proba(X_test)[:, 1]
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_cm = confusion_matrix(y_test, y_pred_rf)

# Importancia de características
feature_imp_df = pd.DataFrame(zip(X_train.columns, best_rf.feature_importances_), 
                              columns=['Feature', 'Importance'])
print("Top 5 características de Random Forest:")
print(feature_imp_df.sort_values('Importance', ascending=False).head(5))

# Gráficos de interacción entre variables predictorias
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_train['creditLimit'], X_train['age'], c=y_train, cmap='bwr', alpha=0.5)
plt.xlabel('Credit Limit')
plt.ylabel('Age')
plt.title('Credit Limit vs Age')
plt.colorbar(label='Default (0 = No, 1 = Yes)')

plt.subplot(1, 3, 2)
plt.scatter(X_train['billAmt1'], X_train['billAmt2'], c=y_train, cmap='bwr', alpha=0.5)
plt.xlabel('Bill Amount 1')
plt.ylabel('Bill Amount 2')
plt.title('BillAmt1 vs BillAmt2')
plt.colorbar(label='Default (0 = No, 1 = Yes)')

plt.subplot(1, 3, 3)
plt.scatter(X_train['nDelay'], X_train['creditLimit'], c=y_train, cmap='bwr', alpha=0.5)
plt.xlabel('Number of Delays')
plt.ylabel('Credit Limit')
plt.title('nDelay vs Credit Limit')
plt.colorbar(label='Default (0 = No, 1 = Yes)')

plt.tight_layout()
plt.show()

# Curva AUC-ROC
plt.figure(figsize=(10, 8))

# Regresión Logística
fpr_log, tpr_log, thresholds_log = roc_curve(y_test, y_prob_log)
roc_auc_log = auc(fpr_log, tpr_log)
gmeans_log = np.sqrt(tpr_log * (1 - fpr_log))
ix_log = np.argmax(gmeans_log)
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {roc_auc_log:.2f})')
plt.scatter(fpr_log[ix_log], tpr_log[ix_log], marker='o', color='red', 
            label=f'Best Threshold = {thresholds_log[ix_log]:.2f}')

# Random Forest
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
gmeans_rf = np.sqrt(tpr_rf * (1 - fpr_rf))
ix_rf = np.argmax(gmeans_rf)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.scatter(fpr_rf[ix_rf], tpr_rf[ix_rf], marker='o', color='green', 
            label=f'Best Threshold = {thresholds_rf[ix_rf]:.2f}')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Imprimir mejores umbrales
print(f"Mejor umbral para Regresión Logística (G-mean = {gmeans_log[ix_log]:.3f}): {thresholds_log[ix_log]:.3f}")
print(f"Mejor umbral para Random Forest (G-mean = {gmeans_rf[ix_rf]:.3f}): {thresholds_rf[ix_rf]:.3f}")

# Matrices de confusión y precisión del modelo
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Matriz de confusión Regresión Logística
sns.heatmap(log_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title(f'Logistic Regression\nAccuracy: {log_accuracy:.4f}')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Matriz de confusión Random Forest
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title(f'Random Forest\nAccuracy: {rf_accuracy:.4f}')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Resultados
print("=== Evaluación de Modelos ===")
print("Regresión Logística:")
print(f"  Precisión del modelo: {log_accuracy:.4f}")
print(f"  AUC: {roc_auc_log:.4f}")
print(f"  Matriz de Confusión:\n{log_cm}")
print("\nRandom Forest:")
print(f"  Precisión del modelo: {rf_accuracy:.4f}")
print(f"  AUC: {roc_auc_rf:.4f}")
print(f"  Matriz de Confusión:\n{rf_cm}")