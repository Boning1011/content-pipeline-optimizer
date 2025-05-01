import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ----- HELPER FUNCTIONS -----
def create_pipeline_diagram(num_tasks, m0, m1, m2):
    fig = go.Figure()
    
    # Add nodes (tasks)
    for i in range(num_tasks):
        fig.add_trace(go.Scatter(
            x=[i], 
            y=[0], 
            mode='markers+text',
            marker=dict(size=30, color='blue'),
            text=[f"T{i+1}"],
            textposition="middle center",
            name=f"Task {i+1}"
        ))
    
    # Add forward arrows
    for i in range(num_tasks-1):
        fig.add_annotation(
            x=i, y=0,
            ax=i+1, ay=0,
            xref="x", yref="y",
            axref="x", ayref="y",
            text=f"{1-m0-m1-m2:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="green",
        )
    
    # Add self-loop arrows (m1)
    for i in range(num_tasks):
        fig.add_annotation(
            x=i+0.2, y=0.15,
            ax=i-0.2, ay=0.15,
            xref="x", yref="y",
            axref="x", ayref="y",
            text=f"m1={m1:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="orange",
            arrowside="end",
            standoff=5
        )
    
    # Add back-1 arrows (m2)
    for i in range(1, num_tasks):
        fig.add_annotation(
            x=i, y=-0.1,
            ax=i-1, ay=-0.1,
            xref="x", yref="y",
            axref="x", ayref="y",
            text=f"m2={m2:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="red",
        )
    
    # Add back-to-beginning arrows (m0)
    for i in range(1, num_tasks):
        y_offset = -0.25 - 0.05 * (i % 3)
        fig.add_annotation(
            x=i, y=y_offset,
            ax=0, ay=y_offset,
            xref="x", yref="y",
            axref="x", ayref="y",
            text=f"m0={m0:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="purple",
        )
    
    # Configure layout with theme-specific settings
    theme_settings = {
        'light': {
            'bg_color': 'white',
            'text_color': 'black',
            'grid_color': 'lightgray'
        },
        'dark': {
            'bg_color': '#0E1117',
            'text_color': 'white',
            'grid_color': '#333'
        }
    }
    
    current_theme = theme_settings['light' if st.session_state.theme == 'light' else 'dark']
    
    fig.update_layout(
        xaxis=dict(
            range=[-0.5, num_tasks-0.5], 
            showticklabels=False,
            showgrid=True,
            gridcolor=current_theme['grid_color']
        ),
        yaxis=dict(
            range=[-0.5, 0.5], 
            showticklabels=False,
            showgrid=True,
            gridcolor=current_theme['grid_color']
        ),
        showlegend=False,
        title="Pipeline Flow Diagram",
        plot_bgcolor=current_theme['bg_color'],
        paper_bgcolor=current_theme['bg_color'],
        font=dict(color=current_theme['text_color']),
        height=400
    )
    
    return fig

def create_transition_matrix_heatmap(Q_full, num_tasks):
    state_labels = [f"T{i+1}" for i in range(num_tasks)] + ["End"]
    
    theme_settings = {
        'light': {
            'bg_color': 'white',
            'text_color': 'black',
            'grid_color': 'lightgray'
        },
        'dark': {
            'bg_color': '#0E1117',
            'text_color': 'white',
            'grid_color': '#333'
        }
    }
    
    current_theme = theme_settings['light' if st.session_state.theme == 'light' else 'dark']
    
    fig = px.imshow(
        Q_full,
        color_continuous_scale='Viridis',
        labels=dict(x="To State", y="From State", color="Probability"),
        title="Transition Matrix (Q)",
        text_auto='.2f'
    )
    
    fig.update_layout(
        plot_bgcolor=current_theme['bg_color'],
        paper_bgcolor=current_theme['bg_color'],
        font=dict(color=current_theme['text_color'])
    )
    
    fig.update_xaxes(
        tickvals=list(range(num_tasks+1)),
        ticktext=state_labels,
        gridcolor=current_theme['grid_color']
    )
    fig.update_yaxes(
        tickvals=list(range(num_tasks+1)),
        ticktext=state_labels,
        gridcolor=current_theme['grid_color']
    )
    
    return fig

def create_line_plot(data, x_col, y_col, title, x_label, y_label):
    theme_settings = {
        'light': {
            'bg_color': 'white',
            'text_color': 'black',
            'grid_color': 'lightgray'
        },
        'dark': {
            'bg_color': '#0E1117',
            'text_color': 'white',
            'grid_color': '#333'
        }
    }
    
    current_theme = theme_settings['light' if st.session_state.theme == 'light' else 'dark']
    
    fig = px.line(
        data,
        x=x_col,
        y=y_col,
        title=title,
        markers=True
    )
    
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        plot_bgcolor=current_theme['bg_color'],
        paper_bgcolor=current_theme['bg_color'],
        font=dict(color=current_theme['text_color']),
        xaxis=dict(gridcolor=current_theme['grid_color']),
        yaxis=dict(gridcolor=current_theme['grid_color'])
    )
    
    return fig

# ----- MARKOV MODEL FUNCTIONS -----
def calculate_expected_time_markov(num_tasks, m0, m1, m2, task_times=None):
    """Calculate expected time for a pipeline using Markov matrix approach
    
    Args:
        num_tasks: Number of tasks in the pipeline
        m0: Jump to beginning mistake rate (0-1)
        m1: Redo current task mistake rate (0-1)
        m2: Jump back one step mistake rate (0-1)
        task_times: List of time costs for each task (default: all 1)
    
    Returns:
        Expected total time
    """
    if task_times is None:
        task_times = [1] * num_tasks
        
    if num_tasks == 0:
        return 0
        
    # Handle edge cases
    if m0 < 0 or m1 < 0 or m2 < 0:
        return float('inf')
    
    if m0 + m1 + m2 >= 1:
        return float('inf')  # Will never complete
    
    # Special case: no mistakes
    if m0 == 0 and m1 == 0 and m2 == 0:
        return sum(task_times)
    
    # Create transition matrix P for all states including absorbing state
    n = num_tasks + 1  # Add absorbing state
    P = np.zeros((n, n))
    
    # Forward probability
    forward_prob = 1 - m0 - m1 - m2
    
    for i in range(num_tasks):
        # Probability of redoing current task
        P[i, i] = m1
        
        # Probability of going back to beginning
        if i > 0:
            P[i, 0] = m0
        
        # Probability of going back one step
        if i > 0:
            P[i, i-1] = m2
        else:
            # If at beginning, going back means staying
            P[i, i] += m2
        
        # Probability of moving forward
        if i < num_tasks - 1:
            P[i, i+1] = forward_prob
        else:
            # From last task, forward means reaching absorbing state
            P[i, -1] = forward_prob
            
        # Ensure row sums to 1 for numerical stability
        row_sum = np.sum(P[i, :])
        if not np.isclose(row_sum, 1.0, rtol=1e-10):
            P[i, :] /= row_sum
    
    # Absorbing state transitions to itself with probability 1
    P[-1, -1] = 1.0
    
    # Extract Q (transitions between non-absorbing states)
    Q = P[:-1, :-1]
    
    # Calculate fundamental matrix N = (I - Q)^-1
    try:
        I = np.eye(num_tasks)
        N = np.linalg.inv(I - Q)
        
        # Check for numerical stability with more lenient bounds
        if np.any(N < -1e-10):  # Allow for small negative values due to numerical error
            return float('inf')
            
        if np.any(N > 1e10):  # More reasonable upper bound
            return float('inf')
            
    except np.linalg.LinAlgError:
        return float('inf')
    
    # Calculate expected time
    t = np.array(task_times)
    E = N @ t  # Matrix multiplication
    
    # Verify result is reasonable with more lenient bounds
    min_time = sum(task_times)  # Perfect execution time
    if E[0] < min_time * 0.999:  # Allow for small numerical errors
        return float('inf')
        
    # Add upper bound check
    if E[0] > min_time * 1000:  # Arbitrary large multiplier
        return float('inf')
    
    return float(E[0])  # Ensure we return a float, not a numpy type

def perform_sensitivity_analysis(num_tasks, base_m0, base_m1, base_m2, param_range, param_type='m1', task_times=None):
    """Perform sensitivity analysis by varying one parameter
    
    Args:
        num_tasks: Number of tasks in the pipeline
        base_m0: Base rate for mistakes requiring going to beginning
        base_m1: Base rate for mistakes requiring redoing current task
        base_m2: Base rate for mistakes requiring going back 1 step
        param_range: Range of parameter values to test
        param_type: Type of parameter to vary ('m0', 'm1', or 'm2')
        task_times: List of time costs for each task (default: all 1)
    
    Returns:
        DataFrame with parameter values and resulting expected times
    """
    results = []
    
    for value in param_range:
        if param_type == 'm0':
            m0 = value
            m1 = base_m1
            m2 = base_m2
        elif param_type == 'm1':
            m0 = base_m0
            m1 = value
            m2 = base_m2
        else:  # 'm2'
            m0 = base_m0
            m1 = base_m1
            m2 = value
        
        # Skip invalid combinations
        if m0 + m1 + m2 >= 1:
            continue
            
        expected_time = calculate_expected_time_markov(num_tasks, m0, m1, m2, task_times)
        
        # Only include finite values
        if not np.isinf(expected_time):
            results.append({
                'Parameter Value': value,
                'Expected Time': expected_time
            })
    
    return pd.DataFrame(results)

def analyze_task_number_impact(base_m0, base_m1, base_m2, min_tasks=2, max_tasks=20):
    """Analyze how efficiency changes with increasing number of tasks
    
    Args:
        base_m0: Base rate for mistakes requiring going to beginning
        base_m1: Base rate for mistakes requiring redoing current task
        base_m2: Base rate for mistakes requiring going back 1 step
        min_tasks: Minimum number of tasks to analyze
        max_tasks: Maximum number of tasks to analyze
    
    Returns:
        DataFrame with tasks count and efficiency metrics
    """
    results = []
    
    for num_tasks in range(min_tasks, max_tasks + 1):
        # Calculate expected time
        expected_time = calculate_expected_time_markov(num_tasks, base_m0, base_m1, base_m2)
        
        # Calculate raw time (ideal case without mistakes)
        raw_time = num_tasks  # Assuming unit time cost
        
        # Calculate efficiency ratio (expected time / raw time)
        if np.isinf(expected_time):
            efficiency_ratio = float('inf')
        else:
            efficiency_ratio = expected_time / raw_time
        
        # Calculate overhead
        if np.isinf(expected_time):
            overhead_percent = float('inf')
        else:
            overhead_percent = (efficiency_ratio - 1) * 100
        
        results.append({
            'Number of Tasks': num_tasks,
            'Expected Time': expected_time,
            'Raw Time': raw_time,
            'Efficiency Ratio': efficiency_ratio,
            'Overhead %': overhead_percent
        })
    
    return pd.DataFrame(results)

# ----- APP INTERFACE -----
def main():
    # Set page config and theme
    st.set_page_config(
        page_title="Markov Matrix Pipeline Simulator",
        layout="wide"
    )

    # Theme state management
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'

    # Theme toggle in sidebar
    with st.sidebar:
        if st.button('Toggle Theme 🌓'):
            st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

    # Custom CSS for theme
    if st.session_state.theme == 'light':
        st.markdown("""
            <style>
            .stApp {
                background-color: white;
                color: black;
            }
            .st-emotion-cache-1y4p8pa {
                padding: 2rem 1rem;
            }
            .st-emotion-cache-16txtl3 {
                padding: 1rem;
            }
            div[data-testid="stPlotlyChart"] {
                background-color: white;
            }
            h1, h2, h3, h4, h5, h6 {
                color: black !important;
            }
            .st-emotion-cache-10trblm {
                color: black !important;
            }
            .st-emotion-cache-1629p8f h1 {
                color: black !important;
            }
            .st-emotion-cache-1629p8f h2 {
                color: black !important;
            }
            .st-emotion-cache-1629p8f h3 {
                color: black !important;
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .stApp {
                background-color: #0E1117;
                color: white;
            }
            </style>
        """, unsafe_allow_html=True)

    st.title("Markov Matrix Pipeline Simulator")
    st.write("""
    This app simulates the expected time to complete a pipeline of media production tasks,
    taking into account three types of mistakes:
    - Type 0 (m0): Mistakes that require going back to the beginning
    - Type 1 (m1): Mistakes that require redoing the current task
    - Type 2 (m2): Mistakes that require going back one step
    """)
    
    # Sidebar for inputs
    st.sidebar.header("Pipeline Parameters")
    
    # Basic parameters
    num_tasks = st.sidebar.slider("Number of Tasks", 3, 15, 5)
    
    # Mistake rates
    st.sidebar.subheader("Mistake Rates")
    m0 = st.sidebar.slider("Type 0 Mistake Rate (m0)", 0.0, 1.0, 0.02, 0.01, 
                          help="Probability of mistakes requiring going back to beginning")
    m1 = st.sidebar.slider("Type 1 Mistake Rate (m1)", 0.0, 1.0, 0.1, 0.01, 
                          help="Probability of mistakes requiring redoing the current task")
    m2 = st.sidebar.slider("Type 2 Mistake Rate (m2)", 0.0, 1.0, 0.05, 0.01,
                          help="Probability of mistakes requiring going back 1 step")
    
    # Normalize rates if their sum exceeds 1
    total_rate = m0 + m1 + m2
    if total_rate > 1:
        st.sidebar.warning(f"Total mistake rate ({total_rate:.2f}) exceeds 1.0. Normalizing rates...")
        m0 = m0 / total_rate
        m1 = m1 / total_rate
        m2 = m2 / total_rate
        st.sidebar.info(f"Normalized rates: m0={m0:.3f}, m1={m1:.3f}, m2={m2:.3f}")
    
    # Display total mistake rate
    st.sidebar.write(f"Total mistake rate: {(m0 + m1 + m2):.3f}")
    
    # Check if rates are valid (should always be true now due to normalization)
    valid_rates = True  # Remove the old validation check since we normalize instead
    
    # Advanced options
    st.sidebar.subheader("Advanced Options")
    custom_times = st.sidebar.checkbox("Use custom task times")
    
    task_times = None
    if custom_times:
        st.sidebar.write("Time units for each task:")
        task_times = []
        
        for i in range(num_tasks):
            time = st.sidebar.number_input(
                f"Task {i+1} time", 
                min_value=0.1, 
                max_value=10.0, 
                value=1.0, 
                step=0.1,
                key=f"time_{i}"
            )
            task_times.append(time)
    
    # Main panel calculations and visualizations
    st.header("Pipeline Analysis Results")
    
    if valid_rates:
        # Calculate expected time using Markov approach
        expected_time = calculate_expected_time_markov(num_tasks, m0, m1, m2, task_times)
        
        # Calculate raw time (sum of all task times)
        raw_time = sum(task_times) if task_times else num_tasks
        
        # Calculate efficiency ratio
        if np.isinf(expected_time):
            efficiency_ratio = float('inf')
            overhead_percent = float('inf')
        else:
            efficiency_ratio = expected_time / raw_time
            overhead_percent = (efficiency_ratio - 1) * 100
        
        # Display results
        st.subheader("Expected Completion Time")
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if np.isinf(expected_time):
                st.error("∞ time units")
                st.caption("Pipeline will never complete with these parameters")
            else:
                st.metric("Expected Time", f"{expected_time:.2f} units")
        
        with col2:
            st.metric("Raw Time", f"{raw_time:.2f} units")
            st.caption("Ideal time without mistakes")
        
        with col3:
            if np.isinf(efficiency_ratio):
                st.error("∞ ratio")
            else:
                st.metric("Efficiency Ratio", f"{efficiency_ratio:.2f}x")
                st.caption(f"+{overhead_percent:.1f}% overhead")
        
        # Visualization of the pipeline
        st.subheader("Pipeline Visualization")
        
        # Create pipeline diagram
        fig = create_pipeline_diagram(num_tasks, m0, m1, m2)
        
        st.plotly_chart(fig)
        
        # Matrix visualization
        st.subheader("Transition Matrix Visualization")
        
        # Create the transition matrix for visualization
        Q = np.zeros((num_tasks, num_tasks))
        forward_prob = 1 - m0 - m1 - m2
        
        for i in range(num_tasks):
            # Self-loop probability
            Q[i, i] = m1
            
            # Probability of going back to beginning
            Q[i, 0] = m0
            
            # Probability of going back one step
            if i > 0:
                Q[i, i-1] = m2
            else:
                # If at beginning, going back means staying
                Q[i, i] += m2
            
            # Probability of moving forward
            if i < num_tasks - 1:
                Q[i, i+1] = forward_prob
        
        # Add absorbing state transition
        Q_full = np.zeros((num_tasks+1, num_tasks+1))
        Q_full[:num_tasks, :num_tasks] = Q
        Q_full[num_tasks, num_tasks] = 1.0  # Absorbing state
        
        # Create heatmap
        fig = create_transition_matrix_heatmap(Q_full, num_tasks)
        
        st.plotly_chart(fig)
        
        # Task number impact analysis
        st.subheader("Impact of Task Count")
        
        # Perform analysis
        task_impact_data = analyze_task_number_impact(m0, m1, m2, min_tasks=2, max_tasks=20)
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Efficiency Ratio", "Raw Cost", "Data"])
        
        with tab1:
            # Create chart for efficiency vs task count
            fig = create_line_plot(
                task_impact_data, 
                "Number of Tasks", 
                "Efficiency Ratio",
                "How Efficiency Ratio Changes with Pipeline Length",
                "Number of Tasks",
                "Efficiency Ratio (Expected Time / Raw Time)"
            )
            
            # Add line for current number of tasks
            fig.add_vline(x=num_tasks, line_dash="dash", line_color="red")
            
            # Highlight current task count efficiency
            current_efficiency = task_impact_data[task_impact_data["Number of Tasks"] == num_tasks]["Efficiency Ratio"].values
            if len(current_efficiency) > 0 and not np.isinf(current_efficiency[0]):
                fig.add_trace(go.Scatter(
                    x=[num_tasks],
                    y=[current_efficiency[0]],
                    mode="markers",
                    marker=dict(color="red", size=12),
                    name=f"Current ({num_tasks} tasks)"
                ))
            
            st.plotly_chart(fig)
            
            # Add log scale option for ratio
            if st.checkbox("Show ratio in logarithmic scale"):
                fig = create_line_plot(
                    task_impact_data, 
                    "Number of Tasks", 
                    "Efficiency Ratio",
                    "Efficiency Ratio (Log Scale)",
                    "Number of Tasks",
                    "Efficiency Ratio (Log Scale)"
                )
                
                fig.update_layout(yaxis_type="log")
                
                # Add line for current number of tasks
                fig.add_vline(x=num_tasks, line_dash="dash", line_color="red")
                
                # Highlight current task count efficiency
                if len(current_efficiency) > 0 and not np.isinf(current_efficiency[0]):
                    fig.add_trace(go.Scatter(
                        x=[num_tasks],
                        y=[current_efficiency[0]],
                        mode="markers",
                        marker=dict(color="red", size=12),
                        name=f"Current ({num_tasks} tasks)"
                    ))
                
                st.plotly_chart(fig)

        with tab2:
            # Create chart for raw expected time vs task count
            fig = create_line_plot(
                task_impact_data, 
                "Number of Tasks", 
                "Expected Time",
                "How Expected Time Changes with Pipeline Length",
                "Number of Tasks",
                "Expected Time (units)"
            )
            
            # Add line for current number of tasks
            fig.add_vline(x=num_tasks, line_dash="dash", line_color="red")
            
            # Add perfect execution line
            fig.add_trace(go.Scatter(
                x=task_impact_data["Number of Tasks"],
                y=task_impact_data["Raw Time"],
                mode="lines",
                line=dict(dash="dot", color="green"),
                name="Perfect Execution (No Mistakes)"
            ))
            
            # Highlight current task count time
            current_time = task_impact_data[task_impact_data["Number of Tasks"] == num_tasks]["Expected Time"].values
            if len(current_time) > 0 and not np.isinf(current_time[0]):
                fig.add_trace(go.Scatter(
                    x=[num_tasks],
                    y=[current_time[0]],
                    mode="markers",
                    marker=dict(color="red", size=12),
                    name=f"Current ({num_tasks} tasks)"
                ))
            
            st.plotly_chart(fig)
            
            # Add log scale option for raw cost
            if st.checkbox("Show cost in logarithmic scale"):
                fig = create_line_plot(
                    task_impact_data, 
                    "Number of Tasks", 
                    "Expected Time",
                    "Expected Time (Log Scale)",
                    "Number of Tasks",
                    "Expected Time (Log Scale)"
                )
                
                fig.update_layout(yaxis_type="log")
                
                # Add line for current number of tasks
                fig.add_vline(x=num_tasks, line_dash="dash", line_color="red")
                
                # Add perfect execution line
                fig.add_trace(go.Scatter(
                    x=task_impact_data["Number of Tasks"],
                    y=task_impact_data["Raw Time"],
                    mode="lines",
                    line=dict(dash="dot", color="green"),
                    name="Perfect Execution (No Mistakes)"
                ))
                
                # Highlight current task count time
                if len(current_time) > 0 and not np.isinf(current_time[0]):
                    fig.add_trace(go.Scatter(
                        x=[num_tasks],
                        y=[current_time[0]],
                        mode="markers",
                        marker=dict(color="red", size=12),
                        name=f"Current ({num_tasks} tasks)"
                    ))
                
                st.plotly_chart(fig)
        
        with tab3:
            # Show data table
            st.dataframe(task_impact_data)
    else:
        st.error("Please adjust mistake rates so their sum is less than 1.0")
        
    # Replace the old expanders with new Mathematical Foundation
    with st.expander("Theoretical Foundation - Markov Chain Model"):
        st.write("""
        ## Mathematical Foundation
        
        This simulator uses an absorbing Markov chain model to calculate the expected completion time of a pipeline with mistakes.
        
        ### State Space
        For a pipeline with n tasks, we define n+1 states:
        - States 0 to n-1: Representing work on each task
        - State n: Absorbing state (pipeline completion)
        
        ### Transition Matrix
        The transition matrix P has dimensions (n+1)×(n+1) where:
        """)
        
        st.latex(r"""
        P = \begin{bmatrix} 
        Q & R \\
        0 & 1
        \end{bmatrix}
        """)
        
        st.write("""
        Where:
        - Q: n×n matrix of transitions between non-absorbing states
        - R: n×1 matrix of transitions to the absorbing state
        - 0: 1×n zero matrix
        - 1: 1×1 identity for the absorbing state
        
        ### Transition Probabilities
        For each state i (0 ≤ i < n):
        """)
        
        st.latex(r"""
        \begin{align*}
        P(i \to i) &= m_1 \text{ (redo current task)} \\
        P(i \to 0) &= m_0 \text{ (return to start)} \\
        P(i \to i-1) &= m_2 \text{ (go back one step)} \\
        P(i \to i+1) &= 1-m_0-m_1-m_2 \text{ (move forward)}
        \end{align*}
        """)
        
        st.write("""
        ### Expected Time Calculation
        The fundamental matrix N gives the expected number of visits to each transient state:
        """)
        
        st.latex(r"""
        N = (I - Q)^{-1}
        """)
        
        st.write("""
        Where:
        - I is the n×n identity matrix
        - Each entry N[i,j] represents the expected number of times the process visits state j when starting from state i
        
        ### Total Expected Time
        If t is the vector of task times, the expected total time E[T] is:
        """)
        
        st.latex(r"""
        E[T] = \mathbf{t}^T \cdot N \cdot \mathbf{1}
        """)
        
        st.write("""
        Where:
        - t is the vector of individual task times
        - 1 is a column vector of ones
        
        ### Convergence Conditions
        The process will converge (finite expected time) if and only if:
        """)
        
        st.latex(r"""
        m_0 + m_1 + m_2 < 1
        """)
        
        st.write("""
        This ensures the spectral radius of Q is less than 1, making (I-Q) invertible.
        
        ### Efficiency Ratio
        The efficiency ratio η is defined as:
        """)
        
        st.latex(r"""
        \eta = \frac{E[T]}{\sum_{i=0}^{n-1} t_i}
        """)
        
        st.write("""
        This ratio compares the expected completion time to the perfect execution time (no mistakes).
        - η = 1: Perfect efficiency (no mistakes)
        - η > 1: Indicates the overhead due to mistakes
        - η → ∞: Process may not complete (mistake rates too high)
        """)

if __name__ == "__main__":
    main()