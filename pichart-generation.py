#Generate a simple pie chart using matplotlib
import matplotlib.pyplot as plt
# Data to plot
labels = 'A', 'B', 'C', 'D'
sizes = [15, 30, 45, 10]
# Colors for each section
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
# Explode the 1st slice (i.e. 'A')
explode = (0.1, 0, 0, 0)
# Create a pie chart
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()