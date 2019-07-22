%% Example application of prinicipal component regression
%%Written by John O'Connor
%%NB all data is simulated and should not be interpreted as otherwise
%%Requires statistics and machine learning toolbox for pca.m function

%% Generate simulated NIRS data

%Seeds random number generation for repeatability
rng(1)

%Generate matrix of zeros for n hypothetical participants
n=1000;
x=zeros(12000,n);

%Generate random start, trough (time and magnitude) and end for stand
start_t=randi([3000,3100],n,1);
trough_t=randi([3200,3400],n,1);
trough_m=(randi([1,10],n,1)*-1);
rec_t=randi([3500,3600],n,1);
            
            for participant=1:n
                
            decline=linspace(0,trough_m(participant), trough_t(participant)-start_t(participant)+1);
            recovery=linspace(trough_m(participant),0, rec_t(participant)-trough_t(participant)+1);

            x(start_t(participant):trough_t(participant),participant)=decline;
            x(trough_t(participant):rec_t(participant),participant)=recovery;

            end

%Add gaussian noise and baseline
x=x+(randn(12000,n))/5;
baseline=repmat(randi([-1 1], n,1),1,12000)';
x=x+baseline;

%Plot first 5 samples
plot(x(:,1:5))

%Plot mean of all data 
plot(mean(x,2))

%%Generate response(y) variable with some assocation to magnitude and timing of drop
y=10*trough_t+10*trough_m+1*baseline(1,:)';
y=y+randn(n,1);

%% Principal component analysis 

[coeff,score,vari,~,exp,mu]=pca(x');

scree=cumsum(exp(1:20));

%%Scree  plot for assessing cumulative variance explained by each component 
plot(1:20, scree,'-ok','LineWidth', 1,'MarkerFaceColor','w');hold on
ylabel('Variance explained (%)'); xlabel('Principal component')
set(gca,'FontSize', 16);
title('Scree plot')

%% Visualising components

%divisions between markers
div=50;
%define time dimension
t=(-60:1/50:180-1/50);

for component=1:6
    
        subplot(3,2,component)
        
        %plot mean
        plot(t,mu, 'black','Color', 'k', 'LineWidth', 1)
        
        %labels
        xlabel('Time (s)')
        ylabel('y')
        set(gca,'FontSize', 16)
        
        %x limits 
        xlim([-10 25])
        hold on; title(['PC' num2str(component)])
        set(gcf,'color','w');  
        
        %upper and lower curves (+/- 2SD)
        down=mu+((-2*sqrt(vari(component,:))).*coeff(:,component))';
        up=mu+((2*sqrt(vari(component,:))).*coeff(:,component))';
                
        plot(t(1:div:end),up(1:div:end),'-ok','LineWidth', 1,'MarkerFaceColor','w','MarkerSize',4)
        plot(t(1:div:end),down(1:div:end),'-^k','LineWidth', 1,'MarkerFaceColor','k','MarkerSize',4)
           
end
legend('Mean', '+2 SD', '-2 SD')

%% Principal component regression 

%%Set up scores as explantory variables 
x1=score(:,1:3);

%%Run linear model and print results
Lmodel = fitlm(x1,y);
Lmodel
