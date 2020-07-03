%% Example application of prinicipal component regression (https://doi.org/10.3389/fnhum.2020.00261)
%%Written by John O'Connor
%%NB all data is simulated and should not be interpreted as otherwise
%%using open FDA software(http://www.psych.mcgill.ca/misc/fda)

clearvars

%% Generate simulated NIRS data

%Seeds random number generation for repeatability
rng(1)

%Generate matrix of zeros for n hypothetical participants
n=1000;
x=zeros(12001,n);

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
x=x+(randn(12001,n))/5;
baseline=repmat(randi([-1 1], n,1),1,12001)';
x=x+baseline;

%Plot first 5 samples
plot(x(:,1:5))

%Plot mean of all data 
plot(mean(x,2))

%%Generate response(y) variable with some assocation to magnitude and timing of drop
y=10*trough_t+10*trough_m+1*baseline(1,:)';
y=y+randn(n,1);

%% Functional principal component analysis

time=(1:12001)';
array(:,:,1)=x;

%% Estimating a functional data object
rng      = [1,12001];
knots=[1 1500 2500:100:5000 8500 12001];
norder   = 6;
nbasis   = length(knots) + norder - 2;
basis_b = create_bspline_basis(rng, nbasis, norder, knots);

%% Linear differential operator 
Lfdobj   = int2Lfd(0);


fdPar_b = fdPar(basis_b,Lfdobj,1e-7);

%% create the fd object

fd = smooth_basis(time,  array, fdPar_b);

%% Generalized cross validation for choosing smoothing parameter

lnlam   = -6:0.25:0;
gcvsave = zeros(length(lnlam),1);
dfsave  = gcvsave;

for i=1:length(lnlam)
  fdPari = fdPar(basis_b, Lfdobj, 10^lnlam(i));
  [fdi, dfi, gcvi] = smooth_basis(time,  array, fdPari);
  gcvsave(i) = sum(gcvi);
  dfsave(i)  = dfi;
end

phdl = plot(lnlam, gcvsave, 'k-o');
set(phdl, 'LineWidth', 2)
xlabel('\fontsize{13} log_{10}(\lambda)')
ylabel('\fontsize{13} GCV(\lambda)')

%% Check fit (one-by-one)
%plotfit_fd(array, time, fd)

%% Perform FPCA
nharm  = 20;
pcastr = pca_fd(fd, nharm, fdPar_b);

%% Scree plot
scree=cumsum((pcastr.varprop(1:end)*100));
plot(1:20, scree,'-ok','LineWidth', 1,'MarkerFaceColor','w');hold on
ylabel('Variance explained (%)'); xlabel('Principal component')
set(gca,'FontSize', 16);
title('Scree plot')

%% Static plots of the principal components
nx=241;
harmfd    = pcastr.harmfd;
fdnames   = getnames(harmfd);        
harmfd    = pcastr.harmfd;
basisobj  = getbasis(harmfd);       
rangex    = getbasisrange(basisobj);
x         = linspace(rangex(1), rangex(2), nx);
meanmat   = squeeze(eval_fd(pcastr.meanfd, x));        
        
div=1;
t=(1:nx)-60;

for i=1:5
        subplot(3,2,i)
        plot(t,meanmat, 'black','Color', 'k', 'LineWidth', 1)
        xlabel('Time (s)')
        ylabel('TSI (%)')
        set(gca,'FontSize', 16)
        
        xlim([-10 25])
        
        hold on; title(['FPC' num2str(i)])
        set(gcf,'color','w');
%         vline(60, '--k','STAND')
        

fdmat     = eval_fd(harmfd, x);
meanmat   = squeeze(eval_fd(pcastr.meanfd, x));
dimfd     = size(fdmat);
nharm     = dimfd(2);
casenames = getfdlabels(fdnames);
fac =2*sqrt(pcastr.values(i));

        vecharm    = fdmat(:,i);
        percentvar = round(100 * pcastr.varprop(i));
        meanplus   = meanmat+fac.*vecharm;
        meanminus  = meanmat-fac.*vecharm;
        
        plot(t(1:div:end),meanplus(1:div:end),'-ok','LineWidth', 1,'MarkerFaceColor','w','MarkerSize',4)
        plot(t(1:div:end),meanminus(1:div:end),'-^k','LineWidth', 1,'MarkerFaceColor','k','MarkerSize',4)
           
end
legend('Mean', '+2 SD', '-2 SD')  

%% Principal component regression 

%%Set up scores as explantory variables 
x1=pcastr.harmscr(:,1:3);

%%Run linear model and print results
Lmodel = fitlm(x1,y);
Lmodel
