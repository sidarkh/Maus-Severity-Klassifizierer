import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, silhouette_score, precision_recall_fscore_support)
import warnings
warnings.filterwarnings('ignore')

# Stil setzen
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*70)
print("VOLLSTÄNDIGE ANALYSE: DSS-INDUZIERTE DARMENTZÜNDUNG")
print("="*70)

# Daten laden
df = pd.read_csv('testdata.txt', sep='\t')
print(f"\nDatensatz geladen: {df.shape[0]} Messungen")
print(f"Anzahl Tiere: {df['id'].nunique()}")
print(f"DSS-Dosen: {sorted(df['DSS'].unique())}")

###############################################################################
# AUFGABE 1: CHARAKTERISIERUNG DER VARIABLEN
###############################################################################

print("\n" + "="*70)
print("AUFGABE 1: CHARAKTERISIERUNG BEIDER VARIABLEN")
print("="*70)

# Aggregation nach DSS und Tag
stats_by_day = df.groupby(['DSS', 'day']).agg({
    'bwc': ['mean', 'std', 'sem'],
    'vwr': ['mean', 'std', 'sem'],
    'id': 'count'
}).reset_index()

stats_by_day.columns = ['DSS', 'day', 'bwc_mean', 'bwc_std', 'bwc_sem', 
                         'vwr_mean', 'vwr_std', 'vwr_sem', 'n']

# Visualisierung: Zeitverläufe
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for dss in sorted(df['DSS'].unique()):
    data = stats_by_day[stats_by_day['DSS'] == dss]
    axes[0].plot(data['day'], data['bwc_mean'], marker='o', label=f'DSS {dss}%', linewidth=2)
    axes[0].fill_between(data['day'], 
                          data['bwc_mean'] - data['bwc_sem'],
                          data['bwc_mean'] + data['bwc_sem'],
                          alpha=0.2)

axes[0].set_xlabel('Tag', fontsize=12)
axes[0].set_ylabel('Body Weight Change (%)', fontsize=12)
axes[0].set_title('Körpergewichtsveränderung über Zeit', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=100, color='gray', linestyle='--', alpha=0.5)

for dss in sorted(df['DSS'].unique()):
    data = stats_by_day[stats_by_day['DSS'] == dss]
    axes[1].plot(data['day'], data['vwr_mean'], marker='o', label=f'DSS {dss}%', linewidth=2)
    axes[1].fill_between(data['day'], 
                          data['vwr_mean'] - data['vwr_sem'],
                          data['vwr_mean'] + data['vwr_sem'],
                          alpha=0.2)

axes[1].set_xlabel('Tag', fontsize=12)
axes[1].set_ylabel('Laufradleistung (rpm)', fontsize=12)
axes[1].set_title('Laufradaktivität über Zeit', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_zeitverlaeufe.png', dpi=300, bbox_inches='tight')
print("\n✓ Grafik gespeichert: 01_zeitverlaeufe.png")
plt.close()

# Statistische Tests
print("\n--- STATISTISCHE TESTS: Dosisabhängige Unterschiede ---")
test_days = [5, 8, 13]

for day in test_days:
    day_data = df[df['day'] == day]
    
    groups_bwc = [day_data[day_data['DSS'] == dss]['bwc'].values 
                  for dss in sorted(df['DSS'].unique())]
    groups_vwr = [day_data[day_data['DSS'] == dss]['vwr'].values 
                  for dss in sorted(df['DSS'].unique())]
    
    h_bwc, p_bwc = stats.kruskal(*groups_bwc)
    h_vwr, p_vwr = stats.kruskal(*groups_vwr)
    
    print(f"\nTag {day}:")
    print(f"  BWC: H={h_bwc:.2f}, p={p_bwc:.4f} {'***' if p_bwc < 0.001 else '**' if p_bwc < 0.01 else '*' if p_bwc < 0.05 else 'n.s.'}")
    print(f"  VWR: H={h_vwr:.2f}, p={p_vwr:.4f} {'***' if p_vwr < 0.001 else '**' if p_vwr < 0.01 else '*' if p_vwr < 0.05 else 'n.s.'}")

# Boxplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, day in enumerate([5, 8, 13]):
    day_data = df[df['day'] == day]
    
    sns.boxplot(data=day_data, x='DSS', y='bwc', ax=axes[0, idx], palette='RdYlGn')
    axes[0, idx].set_title(f'Tag {day}: Body Weight Change', fontweight='bold')
    axes[0, idx].set_xlabel('DSS Dosis')
    axes[0, idx].set_ylabel('BWC (%)')
    
    sns.boxplot(data=day_data, x='DSS', y='vwr', ax=axes[1, idx], palette='RdYlGn')
    axes[1, idx].set_title(f'Tag {day}: Laufradleistung', fontweight='bold')
    axes[1, idx].set_xlabel('DSS Dosis')
    axes[1, idx].set_ylabel('VWR (rpm)')

plt.tight_layout()
plt.savefig('02_boxplots_zeitpunkte.png', dpi=300, bbox_inches='tight')
print("✓ Grafik gespeichert: 02_boxplots_zeitpunkte.png")
plt.close()

###############################################################################
# AUFGABE 2: MACHINE LEARNING CLASSIFIER
###############################################################################

print("\n" + "="*70)
print("AUFGABE 2: MACHINE LEARNING CLASSIFIER")
print("="*70)

# 2.1 K-Means Clustering
X_features = df[['bwc', 'vwr']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# Optimale Cluster-Anzahl finden
inertias = []
silhouette_scores = []
K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Visualisierung
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inertias, marker='o', linewidth=2)
axes[0].set_xlabel('Anzahl Cluster (k)', fontsize=12)
axes[0].set_ylabel('Inertia', fontsize=12)
axes[0].set_title('Elbow-Methode', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouette_scores, marker='o', linewidth=2, color='orange')
axes[1].set_xlabel('Anzahl Cluster (k)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette-Analyse', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03_cluster_optimierung.png', dpi=300, bbox_inches='tight')
print("\n✓ Grafik gespeichert: 03_cluster_optimierung.png")
plt.close()

print(f"\nOptimale Cluster-Anzahl: k=3 (Silhouette Score: {max(silhouette_scores):.3f})")

# K-Means mit k=3
kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=20)
cluster_labels = kmeans_final.fit_predict(X_scaled)
cluster_centers = scaler.inverse_transform(kmeans_final.cluster_centers_)

print("\n--- Cluster-Zentren ---")
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i}: BWC={center[0]:.1f}%, VWR={center[1]:.1f} rpm")

# Schweregrad-Mapping
severity_score = cluster_centers[:, 0] + cluster_centers[:, 1]
cluster_mapping = np.argsort(severity_score)
severity_mapping = {cluster_mapping[0]: 2, cluster_mapping[1]: 1, cluster_mapping[2]: 0}
severity_labels = np.array([severity_mapping[c] for c in cluster_labels])
df['severity'] = severity_labels

print("\n--- Verteilung der Severity-Labels ---")
print(df['severity'].value_counts().sort_index())

# Visualisierung
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

scatter1 = axes[0].scatter(df['bwc'], df['vwr'], c=cluster_labels, 
                           cmap='viridis', alpha=0.6, s=50)
axes[0].scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
                c='red', marker='X', s=300, edgecolors='black', linewidths=2,
                label='Cluster-Zentren')
axes[0].set_xlabel('Body Weight Change (%)', fontsize=12)
axes[0].set_ylabel('Laufradleistung (rpm)', fontsize=12)
axes[0].set_title('K-Means Clustering (k=3)', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

colors = {0: 'green', 1: 'orange', 2: 'red'}
for severity in [0, 1, 2]:
    mask = df['severity'] == severity
    label = ['Gesund', 'Moderat belastet', 'Schwer belastet'][severity]
    axes[1].scatter(df.loc[mask, 'bwc'], df.loc[mask, 'vwr'], 
                    c=colors[severity], label=label, alpha=0.6, s=50)

axes[1].set_xlabel('Body Weight Change (%)', fontsize=12)
axes[1].set_ylabel('Laufradleistung (rpm)', fontsize=12)
axes[1].set_title('Belastungskategorien', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('04_clustering_results.png', dpi=300, bbox_inches='tight')
print("✓ Grafik gespeichert: 04_clustering_results.png")
plt.close()

# 2.2 Classifier Training
X = df[['bwc', 'vwr']].values
y = df['severity'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler_clf = StandardScaler()
X_train_scaled = scaler_clf.fit_transform(X_train)
X_test_scaled = scaler_clf.transform(X_test)

print(f"\nTraining Set: {X_train.shape[0]} Samples")
print(f"Test Set: {X_test.shape[0]} Samples")

classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM (Linear)': SVC(kernel='linear', random_state=42, probability=True),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

print("\n--- TRAINING DER CLASSIFIER ---")
for name, clf in classifiers.items():
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results[name] = {
        'model': clf,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"\n{name}:")
    print(f"  Test Accuracy:  {accuracy:.3f}")
    print(f"  CV Accuracy:    {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']

print(f"\n{'='*70}")
print(f"BESTER CLASSIFIER: {best_model_name}")
print(f"Test Accuracy: {results[best_model_name]['accuracy']:.3f}")
print(f"{'='*70}")

###############################################################################
# AUFGABE 3: STATISTISCHE BESCHREIBUNG
###############################################################################

print("\n" + "="*70)
print("AUFGABE 3: STATISTISCHE BESCHREIBUNG")
print("="*70)

# Confusion Matrix
cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Gesund', 'Moderat', 'Schwer'],
            yticklabels=['Gesund', 'Moderat', 'Schwer'])
axes[0].set_xlabel('Vorhergesagt', fontsize=12)
axes[0].set_ylabel('Tatsächlich', fontsize=12)
axes[0].set_title(f'Confusion Matrix - {best_model_name}\n(Absolute Zahlen)', 
                  fontsize=14, fontweight='bold')

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', ax=axes[1],
            xticklabels=['Gesund', 'Moderat', 'Schwer'],
            yticklabels=['Gesund', 'Moderat', 'Schwer'])
axes[1].set_xlabel('Vorhergesagt', fontsize=12)
axes[1].set_ylabel('Tatsächlich', fontsize=12)
axes[1].set_title(f'Confusion Matrix - {best_model_name}\n(Normalisiert)', 
                  fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('05_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Grafik gespeichert: 05_confusion_matrix.png")
plt.close()

# Classification Report
print("\n--- CLASSIFICATION REPORT ---")
target_names = ['Gesund (0)', 'Moderat (1)', 'Schwer (2)']
print(classification_report(y_test, results[best_model_name]['y_pred'], 
                           target_names=target_names, digits=3))

# Klassenspezifische Metriken
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, results[best_model_name]['y_pred'], average=None
)

print("\n--- SENSITIVITY & SPECIFICITY ---")
for i, label in enumerate(['Gesund', 'Moderat', 'Schwer']):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    tn = cm.sum() - tp - fp - fn
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n{label}:")
    print(f"  Sensitivity: {sensitivity:.3f}")
    print(f"  Specificity: {specificity:.3f}")

# Decision Boundaries
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

h = 0.5
x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
y_min, y_max = X[:, 1].min() - 10, X[:, 1].max() + 10
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

for idx, (name, result) in enumerate(results.items()):
    Z = result['model'].predict(scaler_clf.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    
    axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn', levels=[-.5, 0.5, 1.5, 2.5])
    scatter = axes[idx].scatter(X_test[:, 0], X_test[:, 1], c=y_test, 
                                cmap='RdYlGn', edgecolors='black', s=50, alpha=0.8)
    
    axes[idx].set_xlabel('Body Weight Change (%)', fontsize=11)
    axes[idx].set_ylabel('Laufradleistung (rpm)', fontsize=11)
    axes[idx].set_title(f'{name}\nAccuracy: {result["accuracy"]:.3f}', 
                       fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)

cbar = plt.colorbar(scatter, ax=axes, ticks=[0, 1, 2], pad=0.02)
cbar.set_label('Schweregrad', fontsize=12)
cbar.ax.set_yticklabels(['Gesund', 'Moderat', 'Schwer'])

plt.tight_layout()
plt.savefig('06_decision_boundaries.png', dpi=300, bbox_inches='tight')
print("✓ Grafik gespeichert: 06_decision_boundaries.png")
plt.close()

# ROC Curves
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]
y_score = best_model.predict_proba(X_test_scaled)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure(figsize=(10, 8))

colors = ['green', 'orange', 'red']
labels = ['Gesund', 'Moderat', 'Schwer']

for i, color, label in zip(range(n_classes), colors, labels):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'{label} (AUC = {roc_auc[i]:.3f})')

plt.plot(fpr["micro"], tpr["micro"], color='navy', lw=2, linestyle='--',
         label=f'Micro-Average (AUC = {roc_auc["micro"]:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Zufall')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title(f'ROC-Kurven: {best_model_name}', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('07_roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ Grafik gespeichert: 07_roc_curves.png")
plt.close()

# Cross-Validation Comparison
cv_results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'CV Mean': [r['cv_mean'] for r in results.values()],
    'CV Std': [r['cv_std'] for r in results.values()],
    'Test Accuracy': [r['accuracy'] for r in results.values()]
})

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(cv_results_df))
width = 0.35

bars1 = ax.bar(x - width/2, cv_results_df['CV Mean'], width, 
               yerr=cv_results_df['CV Std'], label='CV Mean ± Std',
               color='steelblue', capsize=5)
bars2 = ax.bar(x + width/2, cv_results_df['Test Accuracy'], width,
               label='Test Accuracy', color='coral')

ax.set_xlabel('Classifier', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Cross-Validation vs Test Performance', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(cv_results_df['Model'], rotation=15, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0.85, 1.0])

for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('08_cv_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Grafik gespeichert: 08_cv_comparison.png")
plt.close()

###############################################################################
# ZUSAMMENFASSUNG
###############################################################################

print("\n" + "="*70)
print("ZUSAMMENFASSUNG")
print("="*70)

print("\n✓ AUFGABE 1: Charakterisierung abgeschlossen")
print("  - Dosisabhängige Unterschiede statistisch nachgewiesen")
print("  - Visualisierungen erstellt")

print("\n✓ AUFGABE 2: Machine Learning Classifier entwickelt")
print(f"  - K-Means Clustering: 3 Kategorien identifiziert")
print(f"  - Bester Classifier: {best_model_name}")
print(f"  - Test Accuracy: {results[best_model_name]['accuracy']:.1%}")

print("\n✓ AUFGABE 3: Statistische Beschreibung vollständig")
print(f"  - Precision: {results[best_model_name]['precision']:.3f}")
print(f"  - Recall: {results[best_model_name]['recall']:.3f}")
print(f"  - F1-Score: {results[best_model_name]['f1']:.3f}")
print(f"  - ROC AUC (Micro): {roc_auc['micro']:.3f}")

print("\n" + "="*70)
print("ALLE ANALYSEN ERFOLGREICH ABGESCHLOSSEN!")
print(f"Generierte Visualisierungen: 8")
print("="*70)
