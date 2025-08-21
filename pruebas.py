V = 2.54
angulo = 180
angulo_60 = 120
V_60 = V*(angulo_60 / angulo)

V5 = 5
V5_bits = 1023
V_60_bits = (V5_bits * V_60) / V5
print("Voltaje a 60 grados:", V_60 )
print("Voltaje a 60 grados en bits:", V_60_bits)