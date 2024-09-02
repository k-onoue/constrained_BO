import torch
import plotly.graph_objects as go


def plot_history(best_values):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(best_values) + 1)),
        y=best_values,
        mode='lines+markers',
        name='Best Value History',
        line=dict(color='blue')
    ))

    fig.update_layout(
        title="Optimization History",
        xaxis_title="Iteration",
        yaxis_title="Best Objective Value"
    )
    
    fig.show()


def plot_best_1D(train_X, train_Y, model, objective_func):
    margin = (train_X.max() - train_X.min()).item() * 0.1
    lower_bound = (train_X.min() - margin).item()
    upper_bound = (train_X.max() + margin).item()
    X = torch.linspace(lower_bound, upper_bound, 100).unsqueeze(-1)
    
    with torch.no_grad():
        posterior = model.posterior(X)
        pred = posterior.mean
        lower, upper = posterior.mvn.confidence_region()

    # 最良の点を見つける
    best_idx = torch.argmax(train_Y)
    best_X = train_X[best_idx].item()
    best_Y = train_Y[best_idx].item()

    fig = go.Figure()
    
    # Objective Function
    fig.add_trace(go.Scatter(
        x=X.numpy().flatten(), 
        y=objective_func(X).numpy().flatten(), 
        mode='lines', 
        name='Objective Function',
        line=dict(color='red')
    ))
    
    # Training Data
    fig.add_trace(go.Scatter(
        x=train_X.numpy().flatten(), 
        y=train_Y.numpy().flatten(), 
        mode='markers', 
        name='Training Data',
        marker=dict(color='black', size=8)
    ))
    
    # GP Prediction
    fig.add_trace(go.Scatter(
        x=X.numpy().flatten(), 
        y=pred.numpy().flatten(), 
        mode='lines', 
        name='GP Prediction',
        line=dict(color='blue')
    ))

    # Confidence Interval
    fig.add_trace(go.Scatter(
        x=X.numpy().flatten(), 
        y=lower.numpy().flatten(), 
        mode='lines', 
        name='Lower Confidence Bound',
        line=dict(color='blue', dash='dash'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=X.numpy().flatten(), 
        y=upper.numpy().flatten(), 
        mode='lines', 
        name='Upper Confidence Bound',
        line=dict(color='blue', dash='dash'),
        fill='tonexty',
        fillcolor='rgba(0, 0, 255, 0.1)',
        showlegend=False
    ))

    # Best Point
    fig.add_trace(go.Scatter(
        x=[best_X],
        y=[best_Y],
        mode='markers',
        name='Best Point',
        marker=dict(color='red', size=20, symbol='star')
    ))

    # # Best Point's vertical and horizontal dashed lines
    # fig.add_shape(type="line",
    #     x0=best_X, y0=0, x1=best_X, y1=best_Y,
    #     line=dict(color="green", width=2, dash="dash")
    # )
    # fig.add_shape(type="line",
    #     x0=0, y0=best_Y, x1=best_X, y1=best_Y,
    #     line=dict(color="green", width=2, dash="dash")
    # )

    fig.update_layout(
        title="Bayesian Optimization Results",
        xaxis_title="X",
        yaxis_title="Objective Value"
    )
    
    fig.show()