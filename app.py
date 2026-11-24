import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt

st.set_page_config(page_title="Ski Telemetry", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""<style>
.stApp { background-color: #0b0f19; color: #f8fafc; }
.block-container { padding-top: 1.2rem; padding-bottom: 4rem; max-width: 520px; }
.card { background: #131a2a; border: 1px solid #1e293b; border-radius: 14px; padding: 14px 16px; margin-bottom: 12px; }
.metric-label { font-size: 12px; color: #94a3b8; margin-bottom: 2px; }
.metric-val { font-size: 18px; font-weight: 700; color: #f8fafc; }
.accent-blue { color: #38bdf8; }
.accent-pink { color: #fb7185; }
div.stButton > button { width: 100%; border-radius: 12px; background: linear-gradient(90deg, #0284c7, #e11d48); color: white; font-weight: 600; border: none; padding: 0.6rem 1rem; }
table.sym-table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; text-align: center; }
table.sym-table th { color: #94a3b8; font-weight: 600; padding-bottom: 6px; border-bottom: 1px solid #1e293b; }
table.sym-table td { padding: 6px 2px; font-weight: 600; }
</style>""", unsafe_allow_html=True)

def process_dataframe(df):
    zc = np.flatnonzero(np.diff(np.sign(df['ang'].to_numpy())) != 0)
    df['turn'] = np.searchsorted(zc, np.arange(len(df)))
    turns = df.groupby('turn').agg(t_start=('t', 'min'), t_end=('t', 'max'), mean_ang=('ang', 'mean'), max_ang=('ang', lambda v: np.max(np.abs(v))), max_g=('g', 'max'), chatter=('chatter', 'mean'), x_mid=('x', 'mean'), y_mid=('y', 'mean')).reset_index()
    turns['time'] = (turns['t_end'] - turns['t_start']).round(2)
    turns = turns[(turns['time'] >= 0.45) & (turns['max_ang'] >= 15)].reset_index(drop=True)
    turns['num'] = turns.index + 1
    work = turns.iloc[1:-1] if len(turns) > 4 else turns
    med_t = work['time'].median()
    turns['delta'] = (turns['time'] - med_t).round(2)
    df_clean = df.merge(turns[['turn', 'num', 'delta']], on='turn', how='inner')
    return {'df': df_clean, 'turns': turns, 'total_time': round(float(df['t'].max()), 1)}

def make_demo_run(seed=1, turns_n=14, speed_mult=1.0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, turns_n * 1.05 * speed_mult, 0.01)
    freq = 1.0 / (2.1 * speed_mult)
    ang = 58.0 * np.sin(2 * np.pi * freq * t) + rng.normal(0, 1.2, len(t))
    err_mask = (t > 6.3 * speed_mult) & (t < 7.6 * speed_mult)
    ang[err_mask] *= 0.85
    g = 1.1 + 1.8 * np.abs(np.sin(2 * np.pi * freq * t)) + rng.normal(0, 0.08, len(t))
    g[err_mask] += 1.4
    chatter = np.abs(rng.normal(1.2, 0.4, len(t)))
    chatter[err_mask] += 5.5
    dt = np.gradient(t)
    heading = np.cumsum(np.sin(np.radians(ang)) * dt * 1.8)
    return process_dataframe(pd.DataFrame({'t': t, 'dt': dt, 'ang': ang, 'g': g, 'chatter': chatter, 'x': np.cumsum(np.sin(heading) * 12.0 * dt), 'y': -np.cumsum(np.cos(heading) * 12.0 * dt)}))

def parse_phyphox(acc_file, gyro_file):
    acc, gyro = pd.read_csv(acc_file), pd.read_csv(gyro_file)
    t = gyro.iloc[:, 0].to_numpy()
    gx, gy, gz = gyro.iloc[:, 1].to_numpy(), gyro.iloc[:, 2].to_numpy(), gyro.iloc[:, 3].to_numpy()
    t_acc = acc.iloc[:, 0].to_numpy()
    ax, ay, az = np.interp(t, t_acc, acc.iloc[:, 1].to_numpy()), np.interp(t, t_acc, acc.iloc[:, 2].to_numpy()), np.interp(t, t_acc, acc.iloc[:, 3].to_numpy())
    fs = 1.0 / np.median(np.diff(t))
    b, a = butter(3, 2.0 / (0.5 * fs), btype='low')
    gyros_f = [filtfilt(b, a, gx), filtfilt(b, a, gy), filtfilt(b, a, gz)]
    roll_axis = int(np.argmax([np.std(g) for g in gyros_f]))
    roll_vel = gyros_f[roll_axis] * (180 / np.pi)
    dt = np.gradient(t)
    b_hp, a_hp = butter(2, 0.15 / (0.5 * fs), btype='high')
    angle = filtfilt(b_hp, a_hp, np.cumsum(roll_vel * dt))
    a_tot = np.sqrt(ax**2 + ay**2 + az**2)
    b_g, a_g = butter(3, 3.0 / (0.5 * fs), btype='low')
    g_force = filtfilt(b_g, a_g, a_tot) / 9.81
    b_ch, a_ch = butter(2, 12.0 / (0.5 * fs), btype='high')
    chatter = np.abs(filtfilt(b_ch, a_ch, a_tot))
    active = np.flatnonzero(np.abs(angle) > 12)
    s_idx, e_idx = max(0, int(active.min() - fs * 0.8)), min(len(t) - 1, int(active.max() + fs * 0.8))
    t_r, ang_r, gf_r, ch_r, dt_r = t[s_idx:e_idx] - t.item(s_idx), angle[s_idx:e_idx], g_force[s_idx:e_idx], chatter[s_idx:e_idx], dt[s_idx:e_idx]
    heading = np.cumsum(np.sin(np.radians(ang_r)) * dt_r * 1.8)
    return process_dataframe(pd.DataFrame({'t': t_r, 'dt': dt_r, 'ang': ang_r, 'g': gf_r, 'chatter': ch_r, 'x': np.cumsum(np.sin(heading) * 12.0 * dt_r), 'y': -np.cumsum(np.cos(heading) * 12.0 * dt_r)}))

if 'sessions' not in st.session_state:
    st.session_state.sessions = [
        {'id': 's1', 'date': '23 ноя', 'disc': 'SL', 'runs': [{'name': 'Заезд №1 (Разминка)', 'data': make_demo_run(10, 14, 1.04)}, {'name': 'Заезд №2 (Боевой)', 'data': make_demo_run(22, 14, 0.96)}]},
        {'id': 's2', 'date': '22 ноя', 'disc': 'GS', 'runs': [{'name': 'Заезд №1', 'data': make_demo_run(35, 10, 1.25)}]}
    ]
st.session_state.setdefault('page', 'home')
st.session_state.setdefault('sel_session_idx', 0)
st.session_state.setdefault('sel_run_idx', 0)

nav1, nav2 = st.columns(2)
if nav1.button("🏠 Главная", key="nav_home"):
    st.session_state.page = 'home'
    st.rerun()
if nav2.button("☰ Заезды тренировки", key="nav_sess"):
    st.session_state.page = 'session'
    st.rerun()

if st.session_state.page == 'home':
    st.markdown("### 🎿 Последние тренировки")
    for idx, sess in enumerate(st.session_state.sessions):
        c1, c2 = st.columns([3, 1.3])
        c1.markdown(f"<div class='card'><b>{sess['date']} &nbsp; <span class='accent-blue'>{sess['disc']}</span></b><br><span style='font-size:12px;color:#94a3b8;'>Заездов: {len(sess['runs'])}</span></div>", unsafe_allow_html=True)
        c2.write("")
        if c2.button("Открыть", key=f"open_s_{idx}"):
            st.session_state.sel_session_idx = idx
            st.session_state.page = 'session'
            st.rerun()
    st.markdown("---")
    st.markdown("### ⚡ Начать новую тренировку")
    disc = st.selectbox("Выберите дисциплину:", ["Произвол", "GS", "SL"])
    date_str = st.text_input("Дата тренировки:", value="24 ноя")
    acc_up = st.file_uploader("Файл Accelerometer.csv", type=['csv'])
    gyro_up = st.file_uploader("Файл Gyroscope.csv", type=['csv'])
    if st.button("Сохранить и анализировать заезд"):
        if acc_up and gyro_up:
            run_data = parse_phyphox(acc_up, gyro_up)
            existing = next((i for i, s in enumerate(st.session_state.sessions) if s['date'] == date_str and s['disc'] == disc), None)
            if existing is not None:
                r_num = len(st.session_state.sessions[existing]['runs']) + 1
                st.session_state.sessions[existing]['runs'].append({'name': f'Заезд №{r_num}', 'data': run_data})
                st.session_state.sel_session_idx, st.session_state.sel_run_idx = existing, r_num - 1
            else:
                st.session_state.sessions.insert(0, {'id': str(len(st.session_state.sessions)+1), 'date': date_str, 'disc': disc, 'runs': [{'name': 'Заезд №1', 'data': run_data}]})
                st.session_state.sel_session_idx, st.session_state.sel_run_idx = 0, 0
            st.session_state.page = 'run'
            st.rerun()
        else:
            st.warning("Загрузи оба файла!")

if st.session_state.page == 'session':
    sess = st.session_state.sessions[st.session_state.sel_session_idx]
    st.markdown(f"### Тренировка: {sess['date']} ({sess['disc']})")
    for r_idx, r in enumerate(sess['runs']):
        d = r['data']
        col_a, col_b = st.columns([3, 1.3])
        col_a.markdown(f"<div class='card'><b>{r['name']}</b><br><span style='font-size:12px;color:#94a3b8;'>Время: {d['total_time']}с | Поворотов: {len(d['turns'])} | Угол: {d['turns']['max_ang'].max():.0f}°</span></div>", unsafe_allow_html=True)
        col_b.write("")
        if col_b.button("Анализ", key=f"run_btn_{r_idx}"):
            st.session_state.sel_run_idx = r_idx
            st.session_state.page = 'run'
            st.rerun()
    if len(sess['runs']) >= 2:
        st.markdown("---")
        st.markdown("#### 🔀 Наложение заездов")
        r_names = [r['name'] for r in sess['runs']]
        c_r1, c_r2 = st.columns(2)
        idx1 = c_r1.selectbox("Заезд 1 (Голубой):", range(len(r_names)), format_func=lambda i: r_names[i], index=0)
        idx2 = c_r2.selectbox("Заезд 2 (Розовый):", range(len(r_names)), format_func=lambda i: r_names[i], index=1)
        d1, d2 = sess['runs'][idx1]['data']['df'], sess['runs'][idx2]['data']['df']
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(x=d1['x'], y=d1['y'], mode='lines', line=dict(color='#38bdf8', width=3), name=r_names[idx1]))
        fig_cmp.add_trace(go.Scatter(x=d2['x'], y=d2['y'], mode='lines', line=dict(color='#fb7185', width=3, dash='dot'), name=r_names[idx2]))
        fig_cmp.update_layout(paper_bgcolor='#131a2a', plot_bgcolor='#131a2a', height=380, margin=dict(t=10, b=10, l=10, r=10), legend=dict(orientation="h", y=1.08, font=dict(color="white")))
        fig_cmp.update_xaxes(scaleanchor="y", scaleratio=1, visible=False, fixedrange=True)
        fig_cmp.update_yaxes(visible=False, fixedrange=True)
        st.plotly_chart(fig_cmp, use_container_width=True, config={'displayModeBar': False})

if st.session_state.page == 'run':
    sess = st.session_state.sessions[st.session_state.sel_session_idx]
    run_obj = sess['runs'][st.session_state.sel_run_idx]
    data, df, turns = run_obj['data'], run_obj['data']['df'], run_obj['data']['turns']
    st.markdown(f"### Анализ · <span class='accent-blue'>{run_obj['name']}</span>", unsafe_allow_html=True)
    col_map, col_info = st.columns([1.05, 1.15])
    pt_colors = np.where(df['delta'] > 0.12, '#fb7185', '#38bdf8')
    fig_run = go.Figure()
fig_run.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='markers+lines', line=dict(color='#334155', width=2), marker=dict(size=5, color=pt_colors), hoverinfo='skip'))
fig_run.add_trace(go.Scatter(x=turns['x_mid'], y=turns['y_mid'], mode='text', text=turns['num'].astype(str), textfont=dict(color='#f8fafc', size=10), hoverinfo='skip'))
fig_run.update_layout(paper_bgcolor='#131a2a', plot_bgcolor='#131a2a', height=410, margin=dict(t=10, b=10, l=5, r=5), showlegend=False)
fig_run.update_xaxes(scaleanchor="y", scaleratio=1, visible=False, fixedrange=True)
fig_run.update_yaxes(visible=False, fixedrange=True)
col_map.plotly_chart(fig_run, use_container_width=True, config={'displayModeBar': False})
col_map.markdown("● Чисто   ● Потеря", unsafe_allow_html=True)
work = turns.iloc[1:-1] if len(turns) > 4 else turns
side_r, side_l = work[work['mean_ang'] > 0], work[work['mean_ang'] < 0]
col_info.markdown(f"Поворотов{len(turns)}Время заезда{data['total_time']} сМакс. угол / G{turns['max_ang'].max():.1f}° / {turns['max_g'].max():.2f}t, с< °GПр{side_r['time'].mean():.2f}{side_r['max_ang'].mean():.0f}°{side_r['max_g'].mean():.2f}Л{side_l['time'].mean():.2f}{side_l['max_ang'].mean():.0f}°{side_l['max_g'].mean():.2f}", unsafe_allow_html=True)
st.markdown("---")
st.markdown("#### 🔍 Детальный разбор поворота")
t_options = [f"Поворот №{int(r.num)} ({r.time:.2f}с | {r.delta:+.2f}с)" for r in turns.itertuples()]
sel_idx = st.selectbox("Выбери поворот:", range(len(t_options)), format_func=lambda i: t_options[i], index=0)
row = turns.iloc[sel_idx]
sub = df[df['turn'] == row['turn']].reset_index(drop=True)
abs_ang = np.abs(sub['ang'].to_numpy())
norm_ang = abs_ang / (np.max(abs_ang) + 1e-6)
d_ang = np.gradient(abs_ang)
p1, p2, p3, p4 = (d_ang >= 0) & (norm_ang < 0.35), (d_ang >= 0) & (norm_ang >= 0.35) & (norm_ang < 0.85), norm_ang >= 0.85, (d_ang < 0) & (norm_ang < 0.85)
t1, t2, t3, t4 = np.sum(sub['dt'].to_numpy()[p1]), np.sum(sub['dt'].to_numpy()[p2]), np.sum(sub['dt'].to_numpy()[p3]), np.sum(sub['dt'].to_numpy()[p4])
tab1, tab2, tab3 = st.tabs(["1. Фазы поворота", "2. Угол (°)", "3. Давление G"])
idx_12, idx_23, idx_34 = int(np.max(np.append(np.flatnonzero(p1), 0))), int(np.max(np.append(np.flatnonzero(p2), 0))), int(np.max(np.append(np.flatnonzero(p3), 0)))
tick_x, tick_y = [], []
dx_a, dy_a = np.gradient(sub['x'].to_numpy()), np.gradient(sub['y'].to_numpy())
for idx in [idx_12, idx_23, idx_34]:
        nrm = np.hypot(dx_a[idx], dy_a[idx]) + 1e-6
        nx, ny = -dy_a[idx] / nrm * 0.4, dx_a[idx] / nrm * 0.4
        tick_x.extend([sub['x'].iloc[idx] - nx, sub['x'].iloc[idx] + nx, None])
        tick_y.extend([sub['y'].iloc[idx] - ny, sub['y'].iloc[idx] + ny, None])
i1, i2, i3, i4 = int(np.median(np.append(np.flatnonzero(p1), 0))), int(np.median(np.append(np.flatnonzero(p2), idx_12))), int(np.median(np.append(np.flatnonzero(p3), idx_23))), int(np.median(np.append(np.flatnonzero(p4), idx_34)))
f1 = go.Figure()
f1.add_trace(go.Scatter(x=sub['x'], y=sub['y'], mode='lines', line=dict(color='#f8fafc', width=5), hoverinfo='skip'))
f1.add_trace(go.Scatter(x=tick_x, y=tick_y, mode='lines', line=dict(color='#38bdf8', width=3), hoverinfo='skip'))
f1.add_trace(go.Scatter(x=[sub['x'].iloc[i1], sub['x'].iloc[i2], sub['x'].iloc[i3], sub['x'].iloc[i4]], y=[sub['y'].iloc[i1], sub['y'].iloc[i2], sub['y'].iloc[i3], sub['y'].iloc[i4]], mode='text', text=[f"  1 ({t1:.2f}с)", f"  2 ({t2:.2f}с)", f"  3 ({t3:.2f}с)", f"  4 ({t4:.2f}с)"], textposition='middle right', textfont=dict(color='#f8fafc', size=13), hoverinfo='skip'))
f1.update_layout(paper_bgcolor='#131a2a', plot_bgcolor='#131a2a', height=300, showlegend=False, margin=dict(t=10, b=10, l=10, r=40))
f1.update_xaxes(scaleanchor="y", scaleratio=1, visible=False, fixedrange=True)
f1.update_yaxes(visible=False, fixedrange=True)
tab1.plotly_chart(f1, use_container_width=True, config={'displayModeBar': False})
f2 = go.Figure()
f2.add_trace(go.Scatter(x=sub['x'], y=sub['y'], mode='markers+lines', line=dict(color='#1e293b', width=6), marker=dict(size=10, color=abs_ang, colorscale=[[0, '#1e293b'], [0.5, '#0284c7'], [1, '#7dd3fc']], cmin=0, cmax=68, colorbar=dict(title="°", len=0.7)), text=[f"Угол: {a:.1f}°" for a in abs_ang], hoverinfo='text'))
f2.update_layout(paper_bgcolor='#131a2a', plot_bgcolor='#131a2a', height=300, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
f2.update_xaxes(scaleanchor="y", scaleratio=1, visible=False, fixedrange=True)
f2.update_yaxes(visible=False, fixedrange=True)
tab2.plotly_chart(f2, use_container_width=True, config={'displayModeBar': False})
f3 = go.Figure()
f3.add_trace(go.Scatter(x=sub['x'], y=sub['y'], mode='markers+lines', line=dict(color='#1e293b', width=6), marker=dict(size=10, color=sub['g'], colorscale=[[0, '#1e293b'], [0.5, '#e11d48'], [1, '#fda4af']], cmin=1.0, cmax=4.5, colorbar=dict(title="G", len=0.7)), text=[f"Давление: {g:.2f} G" for g in sub['g']], hoverinfo='text'))
f3.update_layout(paper_bgcolor='#131a2a', plot_bgcolor='#131a2a', height=300, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
f3.update_xaxes(scaleanchor="y", scaleratio=1, visible=False, fixedrange=True)
f3.update_yaxes(visible=False, fixedrange=True)
tab3.plotly_chart(f3, use_container_width=True, config={'displayModeBar': False})
    advice = "Потеря темпа из-за удара или срыва канта — затянута фаза загрузки." if row['delta'] > 0.12 else "Четкая работа по фазам, раннее давление и быстрый выход из дуги."
    h1 = f"<b>Поворот №{int(row['num'])}</b> | Время: <span class='accent-blue'><b>{row['time']:.2f} с</b></span> ({row['delta']:+.2f} с)<br>"
    h2 = f"Макс. закантовка: <b>{row['max_ang']:.1f}°</b> | Пик G: <span class='accent-pink'><b>{row['max_g']:.2f}</b></span><br>"
    h3 = f"<span style='font-size:12px;color:#94a3b8;'>1-Иниц: {t1:.2f}с · 2-Кант: {t2:.2f}с · 3-Апекс: {t3:.2f}с · 4-Разгр: {t4:.2f}с</span><br><br>"
    h4 = f"💡 <b>Тренер:</b> {advice}"
    st.markdown("<div class='card'>" + h1 + h2 + h3 + h4 + "</div>", unsafe_allow_html=True)


💡 Тренер: {advice}", unsafe_allow_html=True)
