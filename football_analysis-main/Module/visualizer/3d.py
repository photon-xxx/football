import numpy as np
import plotly.graph_objects as go

def plot_3d_pitch_refined(length=105, width=68):
    fig = go.Figure()

    # === 球场底面（浅绿透明） ===
    field_x = [0, length, length, 0]
    field_y = [0, 0, width, width]
    fig.add_trace(go.Mesh3d(
        x=field_x, y=field_y, z=[0, 0, 0, 0],
        color='rgba(0,100,0,0.2)',  # 更高透明度的绿色
        name='Pitch Surface', showscale=False
    ))

    # === 边界线 ===
    fig.add_trace(go.Scatter3d(
        x=[0, length, length, 0, 0],
        y=[0, 0, width, width, 0],
        z=[0]*5, mode='lines',
        line=dict(color='lightgray', width=5),
        name='Boundary'
    ))

    # === 中线 ===
    fig.add_trace(go.Scatter3d(
        x=[length/2, length/2],
        y=[0, width], z=[0, 0],
        mode='lines', line=dict(color='lightgray', width=2),
        name='Midline'
    ))

    # === 中圈 ===
    theta = np.linspace(0, 2*np.pi, 100)
    cx = length/2 + 9.15*np.cos(theta)
    cy = width/2 + 9.15*np.sin(theta)
    fig.add_trace(go.Scatter3d(
        x=cx, y=cy, z=[0]*len(theta),
        mode='lines', line=dict(color='lightgray', width=2),
        name='Center Circle'
    ))

    # === 点球点 ===
    for x0 in [11, length - 11]:
        fig.add_trace(go.Scatter3d(
            x=[x0], y=[width/2], z=[0.1],
            mode='markers', marker=dict(size=4, color='white'),
            name='Penalty Point'
        ))

    # === 禁区 ===
    penalty_length = 16.5
    penalty_width = 40.32
    for x0 in [0, length - penalty_length]:
        fig.add_trace(go.Scatter3d(
            x=[x0, x0, x0 + penalty_length, x0 + penalty_length, x0],
            y=[(width - penalty_width)/2, (width + penalty_width)/2,
               (width + penalty_width)/2, (width - penalty_width)/2,
               (width - penalty_width)/2],
            z=[0]*5, mode='lines',
            line=dict(color='lightgray', width=2),
            name='Penalty Area'
        ))

    # === 球门 ===
    goal_width = 7.32
    goal_height = 2.44
    goal_depth = 2
    for x0 in [0, length]:
        y1 = (width - goal_width) / 2
        y2 = (width + goal_width) / 2
        z = [0, 0, goal_height, goal_height, 0]
        y = [y1, y2, y2, y1, y1]
        x = [x0, x0, x0, x0, x0]
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines', line=dict(color='white', width=3),
            name='Goal'
        ))

    # === 区域划分线（三区 + 通道）===
    thirds = [length/3, 2*length/3]
    for x in thirds:
        fig.add_trace(go.Scatter3d(
            x=[x, x], y=[0, width], z=[0, 0],
            mode='lines', line=dict(color='gray', width=1, dash='dot'),
            name='Zone Line'
        ))

    lanes = [width/4, width/2, 3*width/4]
    for y in lanes:
        fig.add_trace(go.Scatter3d(
            x=[0, length], y=[y, y], z=[0, 0],
            mode='lines', line=dict(color='gray', width=1, dash='dot'),
            name='Lane Line'
        ))

    # === 坐标与背景设置 ===
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='', showgrid=False, showbackground=False,
                       zeroline=False, tickvals=[0, 35, 70, 105],
                       tickfont=dict(color='white')),
            yaxis=dict(title='', showgrid=False, showbackground=False,
                       zeroline=False, tickvals=[0, 34, 68],
                       tickfont=dict(color='white')),
            zaxis=dict(title='', showgrid=False, showbackground=False,
                       zeroline=False, tickvals=[]),
            aspectmode='data',
            bgcolor='rgb(45,45,45)'  # 灰色背景
        ),
        paper_bgcolor='rgb(30,30,30)',
        title=dict(text="⚽ Tactical 3D Pitch (Refined)", font=dict(color='white', size=18)),
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=50)
    )

    fig.show()

plot_3d_pitch_refined()
