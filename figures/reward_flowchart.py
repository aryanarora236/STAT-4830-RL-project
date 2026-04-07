"""Generate reward calculation flowchart for the RL training pipeline."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(1, 1, figsize=(14, 18))
ax.set_xlim(0, 14)
ax.set_ylim(0, 22)
ax.axis('off')

# Colors
BOX_BLUE = '#3498db'
BOX_GREEN = '#2ecc71'
BOX_ORANGE = '#f39c12'
BOX_RED = '#e74c3c'
BOX_GRAY = '#95a5a6'
BOX_PURPLE = '#9b59b6'
LIGHT_BLUE = '#d6eaf8'
LIGHT_GREEN = '#d5f5e3'
LIGHT_ORANGE = '#fdebd0'
LIGHT_RED = '#fadbd8'
LIGHT_GRAY = '#eaeded'
LIGHT_PURPLE = '#e8daef'

def box(x, y, w, h, text, color, textcolor='black', fontsize=9, bold=False):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                     facecolor=color, edgecolor='#2c3e50', linewidth=1.5)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color=textcolor, fontweight=weight, wrap=True,
            multialignment='center')

def arrow(x1, y1, x2, y2, label='', color='#2c3e50', style='->', labelside='right'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=1.5))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        offset = 0.3 if labelside == 'right' else -0.3
        ax.text(mx + offset, my, label, fontsize=7.5, color=color, ha='center', va='center',
                fontstyle='italic')

# Title
ax.text(7, 21.5, 'REINFORCE Reward Calculation Flowchart', ha='center', fontsize=14,
        fontweight='bold', color='#2c3e50')
ax.text(7, 21.1, '(Per batch item in each RL iteration)', ha='center', fontsize=9,
        color='#7f8c8d')

# ===== GENERATION PHASE =====
ax.text(1, 20.5, 'GENERATION PHASE', fontsize=10, fontweight='bold', color=BOX_BLUE)

box(3.5, 19.8, 7, 0.6, 'Generate poker task: (context, question, correct_answer)', LIGHT_BLUE, fontsize=9)
arrow(7, 19.8, 7, 19.4)

box(3.5, 18.8, 7, 0.5, 'Model generates response text', LIGHT_BLUE, fontsize=9)
arrow(7, 19.4, 7, 19.3)
arrow(7, 19.3, 7, 18.8)

# Attempt loop
box(2, 17.7, 10, 0.6, 'Attempt 1/3: Try to extract Python code from response', LIGHT_BLUE, fontsize=9)
arrow(7, 18.8, 7, 18.3)
arrow(7, 18.3, 7, 17.7)

# Decision: code found?
box(2.5, 16.6, 4, 0.7, 'Code found?\n(markdown fences or\npython-like lines)', LIGHT_ORANGE, fontsize=8)
arrow(7, 17.7, 4.5, 17.35)
arrow(4.5, 17.35, 4.5, 16.6)

# YES path
box(8, 16.6, 4, 0.7, 'Execute code in\nsandbox with\nCONTEXT variable', LIGHT_BLUE, fontsize=8)
arrow(6.5, 16.95, 8, 16.95, 'YES', BOX_GREEN)

# NO path - retry
arrow(2.5, 16.95, 1.2, 16.95, 'NO', BOX_RED, labelside='left')
box(0.2, 16.2, 2.2, 1.1, 'Retry with\nerror message\n(up to 3 tries)', LIGHT_RED, fontsize=7.5)
arrow(1.3, 17.3, 1.3, 17.7, '', BOX_RED, '->')

# Exec result
box(8, 15.4, 4, 0.6, 'Execution succeeded?\nHas stdout?', LIGHT_ORANGE, fontsize=8)
arrow(10, 16.6, 10, 15.4)

# YES - got prediction
box(8, 14.3, 4, 0.6, 'predicted = last line\nof stdout', LIGHT_GREEN, fontsize=8, bold=True)
arrow(12, 15.7, 12.5, 15.7, 'YES', BOX_GREEN)
arrow(12.5, 15.7, 12.5, 14.6)
arrow(12.5, 14.6, 12, 14.6)

# NO - exec failed
box(3, 15.0, 4, 0.6, 'All 3 attempts\nfailed?', LIGHT_ORANGE, fontsize=8)
arrow(8, 15.7, 7, 15.7, 'NO', BOX_RED, labelside='left')
arrow(7, 15.7, 7, 15.3)
arrow(7, 15.3, 7, 15.3)
# route to box
arrow(7, 15.3, 5, 15.3)

# 3rd attempt wrap
box(0.3, 14.3, 5, 0.7, 'FALLBACK: parse action word\nfrom raw text, wrap as\nprint("fold") / print("call $X")', LIGHT_RED, fontsize=7.5)
arrow(3, 15.0, 2.8, 14.3)

# Both paths merge
arrow(5.3, 14.6, 5.8, 14.6, '', BOX_GRAY)
arrow(8, 14.6, 7.2, 14.6, '', BOX_GRAY)

# ===== REWARD PHASE =====
ax.text(1, 13.5, 'REWARD CALCULATION', fontsize=10, fontweight='bold', color=BOX_GREEN)

box(2, 12.4, 10, 0.7, 'Parse both actions:\npred_type, pred_amt = parse_action(predicted)\ncorr_type, corr_amt = parse_action(correct_answer)', LIGHT_GREEN, fontsize=8)
arrow(7, 14.3, 7, 13.6)
arrow(7, 13.6, 7, 13.1)
arrow(7, 13.1, 7, 12.4)

# Action match decision
box(2, 11.0, 3.5, 0.7, 'pred_type ==\ncorr_type?', LIGHT_ORANGE, fontsize=8)
arrow(7, 12.4, 3.75, 12.1)
arrow(3.75, 12.1, 3.75, 11.0)

# Exact match
box(0.3, 10.0, 2.5, 0.5, 'reward = 1.0', LIGHT_GREEN, fontsize=9, bold=True)
arrow(2, 11.0, 1.55, 10.5, 'YES', BOX_GREEN, labelside='left')

# Partial match decision
box(5.5, 11.0, 3.5, 0.7, 'Both "staying in"?\n(call/raise/check)', LIGHT_ORANGE, fontsize=8)
arrow(5.5, 11.35, 5.5, 11.35, 'NO', BOX_RED)

# Partial credit
box(5.5, 10.0, 2.2, 0.5, 'reward = 0.3', LIGHT_ORANGE, fontsize=9, bold=True)
arrow(6.6, 11.0, 6.6, 10.5, 'YES', BOX_GREEN)

# Zero
box(9.5, 10.0, 2.5, 0.5, 'reward = 0.0', LIGHT_RED, fontsize=9, bold=True)
arrow(9, 11.35, 9.5, 11.35, 'NO', BOX_RED)
arrow(9.5, 11.35, 10.75, 11.0)
arrow(10.75, 11.0, 10.75, 10.5)

# Examples
ax.text(0.3, 9.3, 'Examples:', fontsize=8, fontweight='bold', color='#2c3e50')
ax.text(0.3, 8.9, r'pred="call \$10", correct="call \$6"  -> type match -> 1.0', fontsize=7.5, color='#27ae60', family='monospace')
ax.text(0.3, 8.6, r'pred="raise \$12", correct="call \$6" -> both staying -> 0.3', fontsize=7.5, color='#f39c12', family='monospace')
ax.text(0.3, 8.3, r'pred="fold", correct="call \$6"      -> fold vs play -> 0.0', fontsize=7.5, color='#e74c3c', family='monospace')
ax.text(0.3, 8.0, r'pred="call \$10", correct="fold"     -> play vs fold -> 0.0', fontsize=7.5, color='#e74c3c', family='monospace')

# ===== REINFORCE UPDATE =====
ax.text(1, 7.3, 'REINFORCE POLICY UPDATE', fontsize=10, fontweight='bold', color=BOX_PURPLE)

box(1, 6.1, 12, 0.7, 'Collect rewards for all 8 batch items: [r1, r2, ..., r8]\navg_reward = mean(rewards)', LIGHT_PURPLE, fontsize=8)
arrow(7, 7.3, 7, 6.8)

box(1, 5.0, 12, 0.6, 'Update baseline (exponential moving average):\nbaseline = 0.95 × baseline + 0.05 × avg_reward', LIGHT_PURPLE, fontsize=8)
arrow(7, 6.1, 7, 5.0)

box(1, 3.9, 12, 0.6, 'Compute advantages: advantage_i = reward_i - baseline\nNormalize: (adv - mean) / std,  then clip to [-2, +2]', LIGHT_PURPLE, fontsize=8)
arrow(7, 5.0, 7, 3.9)

box(1, 2.7, 12, 0.7, 'REINFORCE loss per item:\nloss_i = -advantage_i × log_prob(generated_tokens)\nTotal loss = sum(loss_i) / batch_size', LIGHT_PURPLE, fontsize=8)
arrow(7, 3.9, 7, 2.7)

# Loss sign explanation
box(0.3, 1.4, 6, 0.9, 'advantage > 0 (good action):\nloss > 0 → gradient makes\nthese tokens MORE likely', LIGHT_GREEN, fontsize=7.5)
box(7.3, 1.4, 6, 0.9, 'advantage < 0 (bad action):\nloss < 0 → gradient makes\nthese tokens LESS likely', LIGHT_RED, fontsize=7.5)
arrow(4, 2.7, 3.3, 2.3)
arrow(10, 2.7, 10.3, 2.3)

# Final
box(3, 0.3, 8, 0.6, 'Clip gradients (max_norm=1.0), optimizer.step()', LIGHT_GRAY, fontsize=9, bold=True)
arrow(7, 1.4, 7, 0.9)

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig('figures/reward_flowchart.png', dpi=150, bbox_inches='tight')
print('Saved figures/reward_flowchart.png')
plt.close()
