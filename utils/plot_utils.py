# Plotting utilities
cm2in = 1 / 2.54

a4_papersize_cm = (21.0, 29.7)  #cm
# a4_papersize = a4_papersize_cm * cm2in #cm
margins = 2.54
textwidth_cm = a4_papersize_cm[0] - 2 * margins
textwidth = textwidth_cm * cm2in

figure_sizes = dict()
figure_sizes['standard'] = (textwidth / 2, 7 * cm2in)
print('Plot Utilities')
