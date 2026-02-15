import matplotlib.pyplot as plt

models = ["Speech", "Text", "Fusion"]
accuracy = [90.71, 13.57, 67.86]

plt.figure(figsize=(6,4))
plt.bar(models, accuracy)
plt.ylabel("Accuracy (%)")
plt.title("Emotion Recognition Model Comparison")
plt.savefig("Results/plots/model_comparison.png")
print("Plot saved inside Results/plots/")
