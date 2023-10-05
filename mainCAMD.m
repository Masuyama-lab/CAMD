% 
% (c) 2022 Naoki Masuyama
% 
% CIM-based Adaptive Resonance Theory (ART) for Mixed Data (CAMD) is proposed in:
% 
% N. Masuyama, Y. Nojima, H. Ishibuchi, and Z. Liu, "Adaptive resonance theory-based clustering for handling mixed data," 
% in Proc. of 2022 International Joint Conference on Neural Networks (IJCNN), pp. 1-8, Padua, Italy, July 18-23, 2022.
% https://ieeexplore.ieee.org/document/9892060
% 
% Run "mainCAMD.m"
% 
% Please contact masuyama@omu.ac.jp if you have any problems.
% 



% clc
clear

% rng(1)


% load data
% datalist = {'AcuteInflammations','Statlog_Heart','CreditApproval_removeMissing','German','CMC','Abalone','Adult_removeMissing'};
tmpD = load('dataset/AcuteInflammations');

DATA = tmpD.data;
LABEL = tmpD.target;
attType = tmpD.attType;


% avoid zero label
if size(find(LABEL==0),1) > 0 
    LABEL = LABEL + 1;
end

% avoid 0 value for a categorical variable
% [0 1 2] -> [1 2 3] for avoiding an error
catIdx = find(attType==1);
catD = DATA(:,catIdx);
checkZero = min(catD);
catD(:,checkZero==0) = catD(:,checkZero==0) + 1;
DATA(:,catIdx) = catD;


% Randamize data 
ran = randperm(size(DATA,1));
DATA = DATA(ran,:);
LABEL = LABEL(ran,:);


% Parameters of CAMD =================================================
CAMDnet.numNodes    = 0;    % the number of nodes
CAMDnet.weight      = [];   % node position
CAMDnet.CountNode = [];     % winner counter for each node
CAMDnet.adaptiveSig = [];   % kernel bandwidth for CIM in each node
CAMDnet.threshold = [];     % similarlity thresholds
CAMDnet.activeNodeIdx = []; % nodes for SigmaEstimation
CAMDnet.CountLabel = [];    % counter for labels of each node

CAMDnet.countCategory = {}; % counter for each category on a categorical attribute
DATAcat = DATA(:,attType==1);
for k = 1:size(DATAcat,2)
    CAMDnet.countCategory{k} = zeros(1, max(unique(DATAcat(:,k))));
end
CAMDnet.InitCountCategory = CAMDnet.countCategory; % for initialization

CAMDnet.Lambda = 4;        % an interval for calculating a kernel bandwidth for CIM
% ====================================================================


time_ca_train = 0;

% Train
tic
CAMDnet = CAMD_Train(DATA, LABEL, max(LABEL), CAMDnet, attType);
time_ca_train = time_ca_train + toc;

% Test
[NMI, ARI] = CAMD_Test(DATA, LABEL, CAMDnet, attType);

% Results
disp(['# of Data: ', num2str(size(DATA,1)),', # of Class: ',num2str(max(LABEL))]);
disp(['# of Nodes in CAMD: ', num2str(CAMDnet.numNodes)]);
disp(['NMI: ', num2str(NMI), ', ARI: ',num2str(ARI), ', Training Time: ',num2str(time_ca_train)]);
disp(' ');


