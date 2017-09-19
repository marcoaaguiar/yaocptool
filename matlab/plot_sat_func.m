
f_max = 1 ;
f_min = -1;
f = @(x) f_max - (f_max-f_min)./(1 +  exp(4*x/(f_max-f_min)));

figure('units','centimeters','position',[5 5 12 8]) 

plot(-10:.1:10, f(-10:.1:10),'LineWidth',2)
grid on

xlabel('\xi') % x-axis label
ylabel('\psi') % y-axis label

export_fig sat_function.pdf -transparent  -pdf 
