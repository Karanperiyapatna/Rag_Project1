from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

c = canvas.Canvas("sample.pdf", pagesize=letter)

# Page 1
c.drawString(100, 700, "Artificial Intelligence (AI) is the simulation of human intelligence in machines.")
c.drawString(100, 680, "It enables systems to learn, reason, and make decisions.")
c.drawString(100, 660, "Machine Learning is a subset of AI that allows systems to learn from data.")
c.showPage()

# Page 2
c.drawString(100, 700, "Deep Learning is a subset of Machine Learning using neural networks.")
c.drawString(100, 680, "AI is used in healthcare, finance, and automation.")
c.save()
