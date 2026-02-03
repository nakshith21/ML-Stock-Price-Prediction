import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PART 1: FEATURE ENGINEERING
# ============================================

def create_features(df):
    """
    Create powerful features for ML model
    
    Feature Categories:
    1. Lagged Returns (past price changes)
    2. Volume Ratios (trading activity)
    3. Volatility Measures (risk metrics)
    4. Technical Indicators (momentum, trend)
    """
    
    print("\n🔧 Creating features...")
    
    # Make a copy
    data = df.copy()
    
    # ========================================
    # 1. LAGGED RETURNS (Look back at past)
    # ========================================
    # "How much did price change in last N days?"
    
    data['return_1d'] = data['Close'].pct_change(1)  # Yesterday's return
    data['return_2d'] = data['Close'].pct_change(2)  # 2 days ago
    data['return_5d'] = data['Close'].pct_change(5)  # 5 days ago (1 week)
    data['return_10d'] = data['Close'].pct_change(10)  # 10 days ago
    data['return_20d'] = data['Close'].pct_change(20)  # 20 days ago (1 month)
    
    # Cumulative returns (running total)
    data['return_cumsum_5d'] = data['return_1d'].rolling(5).sum()
    data['return_cumsum_10d'] = data['return_1d'].rolling(10).sum()
    
    print("   ✅ Created lagged returns (7 features)")
    
    # ========================================
    # 2. VOLUME RATIOS (Trading activity)
    # ========================================
    # "Is volume higher or lower than usual?"
    
    # Volume moving averages
    data['volume_ma5'] = data['Volume'].rolling(5).mean()
    data['volume_ma10'] = data['Volume'].rolling(10).mean()
    data['volume_ma20'] = data['Volume'].rolling(20).mean()
    
    # Volume ratios (current vs average)
    data['volume_ratio_5d'] = data['Volume'] / data['volume_ma5']
    data['volume_ratio_10d'] = data['Volume'] / data['volume_ma10']
    data['volume_ratio_20d'] = data['Volume'] / data['volume_ma20']
    
    # Volume change
    data['volume_change'] = data['Volume'].pct_change(1)
    
    print("   ✅ Created volume features (7 features)")
    
    # ========================================
    # 3. VOLATILITY MEASURES (Risk metrics)
    # ========================================
    # "How much is price bouncing around?"
    
    # Standard deviation of returns (classic volatility)
    data['volatility_5d'] = data['return_1d'].rolling(5).std()
    data['volatility_10d'] = data['return_1d'].rolling(10).std()
    data['volatility_20d'] = data['return_1d'].rolling(20).std()
    
    # High-Low range (daily range)
    data['high_low_range'] = (data['High'] - data['Low']) / data['Close']
    
    # ATR (Average True Range) - advanced volatility
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['atr_14'] = true_range.rolling(14).mean()
    
    print("   ✅ Created volatility features (5 features)")
    
    # ========================================
    # 4. TECHNICAL INDICATORS
    # ========================================
    
    # RSI (from previous project!)
    def calculate_rsi(prices, period=14):
        deltas = prices.diff()
        gains = deltas.where(deltas > 0, 0).rolling(period).mean()
        losses = -deltas.where(deltas < 0, 0).rolling(period).mean()
        rs = gains / losses
        return 100 - (100 / (1 + rs))
    
    data['rsi_14'] = calculate_rsi(data['Close'], 14)
    
    # Moving Average Crossovers
    data['ma_5'] = data['Close'].rolling(5).mean()
    data['ma_20'] = data['Close'].rolling(20).mean()
    data['ma_50'] = data['Close'].rolling(50).mean()
    
    # Price vs MA (is price above or below average?)
    data['price_to_ma5'] = data['Close'] / data['ma_5']
    data['price_to_ma20'] = data['Close'] / data['ma_20']
    data['price_to_ma50'] = data['Close'] / data['ma_50']
    
    # MA crossover signals
    data['ma5_ma20_cross'] = (data['ma_5'] > data['ma_20']).astype(int)
    data['ma20_ma50_cross'] = (data['ma_20'] > data['ma_50']).astype(int)
    
    # MACD
    ema_12 = data['Close'].ewm(span=12).mean()
    ema_26 = data['Close'].ewm(span=26).mean()
    data['macd'] = ema_12 - ema_26
    data['macd_signal'] = data['macd'].ewm(span=9).mean()
    data['macd_hist'] = data['macd'] - data['macd_signal']
    
    # Bollinger Bands
    data['bb_middle'] = data['Close'].rolling(20).mean()
    bb_std = data['Close'].rolling(20).std()
    data['bb_upper'] = data['bb_middle'] + (2 * bb_std)
    data['bb_lower'] = data['bb_middle'] - (2 * bb_std)
    data['bb_position'] = (data['Close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
    
    print("   ✅ Created technical indicators (14 features)")
    
    # ========================================
    # 5. TIME-BASED FEATURES
    # ========================================
    
    # Day of week (Monday effect?)
    data['day_of_week'] = data.index.dayofweek
    
    # Month (seasonal patterns?)
    data['month'] = data.index.month
    
    # Quarter
    data['quarter'] = data.index.quarter
    
    print("   ✅ Created time features (3 features)")
    
    # ========================================
    # 6. TARGET VARIABLE
    # ========================================
    # What we're trying to predict: "Will price go UP tomorrow?"
    
    data['future_return'] = data['Close'].pct_change(1).shift(-1)
    data['target'] = (data['future_return'] > 0).astype(int)  # 1 = UP, 0 = DOWN
    
    print("   ✅ Created target variable\n")
    print(f"📊 Total features created: {len(data.columns) - len(df.columns)}")
    
    return data


def select_features(df):
    """
    Select the most important features for ML model
    """
    feature_columns = [
        # Lagged returns
        'return_1d', 'return_2d', 'return_5d', 'return_10d', 'return_20d',
        'return_cumsum_5d', 'return_cumsum_10d',
        
        # Volume
        'volume_ratio_5d', 'volume_ratio_10d', 'volume_ratio_20d',
        'volume_change',
        
        # Volatility
        'volatility_5d', 'volatility_10d', 'volatility_20d',
        'high_low_range', 'atr_14',
        
        # Technical indicators
        'rsi_14', 'price_to_ma5', 'price_to_ma20', 'price_to_ma50',
        'ma5_ma20_cross', 'ma20_ma50_cross',
        'macd', 'macd_signal', 'macd_hist',
        'bb_position',
        
        # Time features
        'day_of_week', 'month', 'quarter'
    ]
    
    # Remove rows with NaN (due to rolling windows)
    df_clean = df[feature_columns + ['target']].dropna()
    
    X = df_clean[feature_columns]
    y = df_clean['target']
    
    return X, y, feature_columns


# ============================================
# PART 2: MACHINE LEARNING MODEL
# ============================================

class MLTradingModel:
    """
    Random Forest model for predicting stock price direction
    """
    
    def __init__(self, n_estimators=100, max_depth=10):
        """
        n_estimators: Number of trees in forest (default 100)
        max_depth: Maximum depth of each tree (default 10)
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        self.feature_names = None
        self.feature_importance = None
    
    
    def train(self, X_train, y_train):
        """Train the Random Forest model"""
        print("\n🤖 Training Random Forest model...")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Features: {X_train.shape[1]}")
        
        self.model.fit(X_train, y_train)
        self.feature_names = X_train.columns.tolist()
        self.feature_importance = self.model.feature_importances_
        
        print("   ✅ Model trained!")
    
    
    def predict(self, X):
        """Predict price direction"""
        return self.model.predict(X)
    
    
    def predict_proba(self, X):
        """Predict probability of UP"""
        return self.model.predict_proba(X)[:, 1]
    
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        print("\n📊 Model Performance on Test Set:")
        print("="*50)
        
        # Predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n✅ Accuracy: {accuracy*100:.2f}%")
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['DOWN (0)', 'UP (1)']))
        
        # ROC-AUC Score
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"📈 ROC-AUC Score: {roc_auc:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
    
    
    def cross_validate(self, X, y, cv=5):
        """
        Time Series Cross-Validation
        
        Important: Regular CV shuffles data, but time series data has order!
        TimeSeriesSplit respects chronological order
        """
        print("\n🔄 Running Time Series Cross-Validation...")
        print(f"   Number of splits: {cv}")
        
        tscv = TimeSeriesSplit(n_splits=cv)
        scores = cross_val_score(self.model, X, y, cv=tscv, scoring='accuracy')
        
        print(f"\n📊 Cross-Validation Results:")
        print(f"   Scores: {[f'{s:.4f}' for s in scores]}")
        print(f"   Mean Accuracy: {scores.mean()*100:.2f}%")
        print(f"   Std Deviation: {scores.std()*100:.2f}%")
        
        return scores
    
    
    def plot_feature_importance(self, top_n=20):
        """Plot most important features"""
        if self.feature_importance is None:
            print("⚠️ Train model first!")
            return
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.feature_importance
        }).sort_values('Importance', ascending=False)
        
        # Plot top N
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['Importance'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Most Important Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("\n📊 Feature importance chart saved as 'feature_importance.png'")
        plt.close()
        
        # Print top features
        print(f"\n🏆 Top {top_n} Most Important Features:")
        print("="*50)
        for i, row in top_features.iterrows():
            print(f"{row['Feature']:30s} {row['Importance']:.4f}")
        
        return importance_df
    
    
    def plot_confusion_matrix(self, cm):
        """Visualize confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['DOWN', 'UP'],
                   yticklabels=['DOWN', 'UP'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("📊 Confusion matrix saved as 'confusion_matrix.png'")
        plt.close()
    
    
    def plot_roc_curve(self, y_test, y_proba):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        print("📊 ROC curve saved as 'roc_curve.png'")
        plt.close()


# ============================================
# PART 3: BACKTESTING WITH ML PREDICTIONS
# ============================================

def backtest_ml_strategy(df, X_test, y_test, y_pred, y_proba, initial_capital=10000):
    """
    Backtest trading strategy using ML predictions
    
    Strategy:
    - Buy if model predicts UP with >60% confidence
    - Sell if model predicts DOWN
    - Hold if uncertain
    """
    print("\n💰 Backtesting ML Trading Strategy...")
    print("="*50)
    
    # Align predictions with test data
    test_dates = X_test.index
    
    # Create backtest dataframe
    backtest_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Probability': y_proba
    }, index=test_dates)
    
    # Get prices for these dates
    backtest_df = backtest_df.join(df[['Close']])
    
    # Generate trading signals
    backtest_df['Signal'] = 0
    backtest_df.loc[backtest_df['Probability'] > 0.6, 'Signal'] = 1  # Buy
    backtest_df.loc[backtest_df['Probability'] < 0.4, 'Signal'] = -1  # Sell
    
    # Calculate returns
    backtest_df['Market_Return'] = backtest_df['Close'].pct_change()
    backtest_df['Strategy_Return'] = backtest_df['Signal'].shift(1) * backtest_df['Market_Return']
    
    # Calculate cumulative returns
    backtest_df['Market_Cumulative'] = (1 + backtest_df['Market_Return']).cumprod()
    backtest_df['Strategy_Cumulative'] = (1 + backtest_df['Strategy_Return'].fillna(0)).cumprod()
    
    # Portfolio values
    backtest_df['Market_Value'] = initial_capital * backtest_df['Market_Cumulative']
    backtest_df['Strategy_Value'] = initial_capital * backtest_df['Strategy_Cumulative']
    
    # Performance metrics
    total_return = (backtest_df['Strategy_Value'].iloc[-1] / initial_capital - 1) * 100
    market_return = (backtest_df['Market_Value'].iloc[-1] / initial_capital - 1) * 100
    
    # Count trades
    trades = (backtest_df['Signal'].diff() != 0).sum()
    
    print(f"\n📈 Results:")
    print(f"   Strategy Return: {total_return:.2f}%")
    print(f"   Buy & Hold Return: {market_return:.2f}%")
    print(f"   Outperformance: {total_return - market_return:.2f}%")
    print(f"   Total Trades: {trades}")
    
    # Plot results
    plt.figure(figsize=(14, 6))
    plt.plot(backtest_df.index, backtest_df['Strategy_Value'], 
            label='ML Strategy', linewidth=2)
    plt.plot(backtest_df.index, backtest_df['Market_Value'], 
            label='Buy & Hold', linewidth=2, linestyle='--')
    plt.axhline(initial_capital, color='gray', linestyle=':', label='Initial Capital')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.title('ML Trading Strategy vs Buy & Hold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('ml_backtest.png', dpi=300, bbox_inches='tight')
    print("📊 Backtest chart saved as 'ml_backtest.png'")
    plt.close()
    
    return backtest_df


# ============================================
# MAIN EXECUTION
# ============================================

def run_ml_pipeline(ticker='AAPL', period='3y', test_size=0.2):
    """
    Complete ML trading pipeline
    """
    import time
    
    print("="*70)
    print(f"MACHINE LEARNING PRICE PREDICTOR: {ticker}")
    print("="*70)
    
    # Step 1: Download data with retry logic
    print(f"\n📥 Downloading {period} of {ticker} data...")
    stock = yf.Ticker(ticker)
    
    max_retries = 3
    df = None
    
    for attempt in range(max_retries):
        try:
            print(f"   Attempt {attempt + 1}/{max_retries}...")
            df = stock.history(period=period)
            if len(df) > 0:
                print(f"✅ Downloaded {len(df)} days of data")
                break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"   ⚠️ Error, retrying in 2 seconds...")
                time.sleep(2)
            else:
                print(f"   ❌ Failed to download data: {e}")
                print("\n💡 Try:")
                print("   1. Check your internet connection")
                print("   2. Try a different stock ticker")
                print("   3. Use a shorter period (e.g., '2y' or '1y')")
                return None, None, None, None, None, None
    
    if df is None or len(df) == 0:
        print("   ❌ No data downloaded!")
        return None, None, None, None, None, None
    
    # Step 2: Create features
    df_features = create_features(df)
    
    # Step 3: Select features and target
    X, y, feature_names = select_features(df_features)
    
    print(f"\n📊 Dataset Info:")
    print(f"   Total samples: {len(X)}")
    print(f"   Features: {len(feature_names)}")
    print(f"   Target distribution:")
    print(f"      UP days: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"      DOWN days: {len(y) - y.sum()} ({(1-y.mean())*100:.1f}%)")
    
    # Step 4: Split data (time series split - don't shuffle!)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n📊 Train/Test Split:")
    print(f"   Training: {len(X_train)} samples ({(1-test_size)*100:.0f}%)")
    print(f"   Testing: {len(X_test)} samples ({test_size*100:.0f}%)")
    
    # Step 5: Train model
    model = MLTradingModel(n_estimators=100, max_depth=10)
    model.train(X_train, y_train)
    
    # Step 6: Cross-validation
    cv_scores = model.cross_validate(X_train, y_train, cv=5)
    
    # Step 7: Evaluate on test set
    results = model.evaluate(X_test, y_test)
    
    # Step 8: Feature importance
    importance_df = model.plot_feature_importance(top_n=20)
    
    # Step 9: Visualizations
    model.plot_confusion_matrix(results['confusion_matrix'])
    model.plot_roc_curve(y_test, results['y_proba'])
    
    # Step 10: Backtest
    backtest_df = backtest_ml_strategy(df, X_test, y_test, 
                                       results['y_pred'], results['y_proba'])
    
    print("\n" + "="*70)
    print("✅ ML PIPELINE COMPLETE!")
    print("="*70)
    
    return model, X_test, y_test, results, backtest_df, importance_df


if __name__ == "__main__":
    
    # Run full pipeline
    model, X_test, y_test, results, backtest_df, importance_df = run_ml_pipeline(
        ticker='AAPL',
        period='3y',
        test_size=0.2
    )
    
    print("\n💡 Try different stocks:")
    print("run_ml_pipeline('TSLA')")
    print("run_ml_pipeline('GOOGL')")
    print("run_ml_pipeline('NVDA')")
    
    print("\n📊 Charts saved:")
    print("   - feature_importance.png")
    print("   - confusion_matrix.png")
    print("   - roc_curve.png")
    print("   - ml_backtest.png")