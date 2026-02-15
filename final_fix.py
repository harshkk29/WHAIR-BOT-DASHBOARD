import os

file_path = 'weather_dashboard.py'
with open(file_path, 'r') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    line_num = i + 1
    if line_num == 918: # '                except Exception as e:'
         new_lines.append("        except Exception as e:\n")
    elif line_num == 919: # '                            st.error(...)'
         new_lines.append("            st.error(f\"WHAIR BOT is currently resting: {e}\")\n")
    elif line_num == 921: # '    else:'
         new_lines.append("else:\n")
    elif line_num == 922: # '        st.error(...)'
         new_lines.append("    st.error(\"Location not found. Please check the City and Country code.\")\n")
    else:
         new_lines.append(line)

with open(file_path, 'w') as f:
    f.writelines(new_lines)
