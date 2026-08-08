import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. डेटा तैयार करना
np.random.seed(42)
data_size = 1000
df = pd.DataFrame({
    'user_id': range(1001, 1001 + data_size),
    'engagement_score': np.random.normal(68, 14, data_size).clip(10, 100),
    'session_duration_min': np.random.exponential(25, data_size) + 2,
    'pages_visited': np.random.poisson(6, data_size) + 1,
    'feature_clicks': np.random.normal(45, 12, data_size).clip(0, 100)
})

# 2. ग्राफ और चार्ट बनाना
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('High-Density User Behavior Telemetry Visualizer', fontsize=18, fontweight='bold', y=0.98)

# Chart 1: Distribution Plot
sns.histplot(df['engagement_score'], kde=True, color='#2b5c8f', ax=axes, edgecolor='black', alpha=0.7)
axes.set_title('A. User Engagement Score Distribution', fontsize=12, fontweight='bold')

# Chart 2: Scatter Plot
sns.scatterplot(x='session_duration_min', y='pages_visited', data=df, color='#d95f02', alpha=0.5, ax=axes, s=40)
axes.set_title('B. Session Duration vs Pages Visited', fontsize=12, fontweight='bold')
axes.set_xlim(0, 150)

# Chart 3: KDE Plot
sns.kdeplot(df['feature_clicks'], fill=True, color='#2ca02c', ax=axes, alpha=0.4)
axes.set_title('C. Feature Clicks Behavioral Pattern (KDE)', fontsize=12, fontweight='bold')

# Chart 4: Correlation Heatmap
corr_matrix = df[['engagement_score', 'session_duration_min', 'pages_visited', 'feature_clicks']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=1, ax=axes)
axes.set_title('D. High-Density Feature Correlation Matrix', fontsize=12, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('telemetry_visualization.png', dpi=300)
plt.show()
