% load('dir_method.mat')
load('aug_lagrange.mat')

figure('units','centimeters','position',[5 5 12 8]) 
grid on
hold on
h1 = plot(T, X(:,1),'--','LineWidth',2);
h2 = plot(T, X(:,3),'-','LineWidth',2);

legend([h1 h2],{'x_1','x_3'});  % Only the blue and green lines appear
xlim([0 10.05])

xlabel('Time (s)') % x-axis label
ylabel('State') % y-axis label

% export_fig states.pdf -pdf -transparent 
export_fig states2.pdf -pdf -transparent 
