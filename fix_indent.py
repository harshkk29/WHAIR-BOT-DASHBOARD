import sys

with open('weather_dashboard.py', 'r') as f:
    lines = f.readlines()

new_lines = []
start_fixing = False
for i, line in enumerate(lines):
    # Lines between 348 (SECTION 1) and 904 (except block)
    # Based on the previous view_file, the indentation error starts around there
    if "if current.get('wind_speed_10m', 0) > 40:" in line and i > 340:
        start_fixing = True
    
    if start_fixing and "st.session_state.messages.append" in line and "assistant" in line:
        # Don't unindent after this
        new_lines.append(line.lstrip('    '))
        start_fixing = False
        continue

    if start_fixing:
        # Check if line is just whitespace
        if line.strip() == "":
            new_lines.append("\n")
        else:
            # Unindent by 1 level (4 spaces or 1 tab)
            if line.startswith('            '):
                 new_lines.append(line[12:]) # It was double-indented or something? 
            elif line.startswith('        '):
                new_lines.append(line[8:])
            elif line.startswith('    '):
                new_lines.append(line[4:])
            else:
                new_lines.append(line)
    else:
        new_lines.append(line)

with open('weather_dashboard_fixed.py', 'w') as f:
    f.writelines(new_lines)
