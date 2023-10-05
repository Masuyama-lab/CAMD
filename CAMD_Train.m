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
function net = CAMD_Train(DATA, LABELS, maxLABEL, net, attType)

numNodes = net.numNodes;         % the number of nodes
weight = net.weight;             % node position
CountNode = net.CountNode;       % winner counter for each node
adaptiveSig = net.adaptiveSig;   % kernel bandwidth for CIM in each node
threshold = net.threshold;
CountLabel = net.CountLabel;
activeNodeIdx = net.activeNodeIdx;

countCategory = net.countCategory;
InitCountCategory = net.InitCountCategory;

Lambda = net.Lambda;             % an interval for calculating a kernel bandwidth for CIM


% Set a size of CountLabel
if size(weight) == 0
    CountLabel = zeros(1, maxLABEL);
end

bufferNode = Lambda;

for sampleNum = 1:size(DATA,1)
    
    % Current data sample.
    input = DATA(sampleNum,:);
    label = LABELS(sampleNum, 1);
    
    if size(weight,1) < bufferNode % In the case of the number of nodes in the entire space is small.
        
        % Generate 1st to bufferNode-th node from inputs.
        if size(DATA,1) < bufferNode
            numInitNode = size(DATA,1);
        else
            numInitNode = bufferNode;
        end
        for k = 1:numInitNode
            numNodes = numNodes + 1;
            CountNode(numNodes) = 1;
            CountLabel(numNodes, LABELS(k, 1)) = 1;
            weight(numNodes,:) = DATA(k,:);
            
            % Update a counter for each category on a categorical attribute
            if numNodes == 1
                for m = 1:size(weight(:,attType==1),2)
                    Wcat = weight(numNodes,attType==1); % Extract categorical attributes
                    att = countCategory{numNodes,m};
                    att(1,Wcat(m)) = att(1,Wcat(m)) + 1; % Update a count of k-th category
                    countCategory{numNodes,m} = att;
                end
            else
                % Update categorical attributes by init weight
                countCategory = [countCategory; InitCountCategory];
                for m = 1:size(weight(:,attType==1),2)
                    Wcat = weight(numNodes,attType==1); % Extract categorical attributes
                    att = countCategory{numNodes,m};
                    att(1,Wcat(m)) = att(1,Wcat(m)) + 1; % Update a count of k-th category
                    countCategory{numNodes,m} = att;
                end
            end
            
        end
        activeNodeIdx = 1:numNodes;
        initSig = SigmaEstimationByNode(weight(:,attType==0), bufferNode, activeNodeIdx);
        adaptiveSig = repmat(initSig,1,numNodes); % Assign the same initSig to the all nodes.
        
        % Calculate the initial similarlity threshold to the initial nodes.
        if numNodes == bufferNode
            tmpTh = zeros(1,bufferNode);
            for k = 1:bufferNode
                
                % Similarity measure for a numerical attribute
                tmpCIMs1 = CIM(weight(k,attType==0), weight(:,attType==0), mean(adaptiveSig));
                [~, a1] = min(tmpCIMs1);
                tmpCIMs1(a1) = []; % Remove a value between weight(k,:) and weight(k,:)
                minTmpCIMs1 = min(tmpCIMs1);
                
                % Similarity measure for a categorical attribute
                tmpDiff = double( ne(weight(k,attType==1), weight(:,attType==1))' ); % A~=B:1, A==B:0
                catDiff = sum(tmpDiff, 1);
                catDiff = catDiff./max(catDiff);
                [~, b1] = min(catDiff);
                catDiff(b1) = []; % Remove a value between weight(k,:) and weight(k,:)
                minCatDiff = min(catDiff);
                
                % Multiply a ratio of the number of attributes
                minTmpCIMs1 = minTmpCIMs1 * size(find(attType==0),2)/size(attType,2);
                minCatDiff = minCatDiff * size(find(attType==1),2)/size(attType,2);
                
                % Sum up the similarity by numerical and categorical attributes
                tmpTh(k) = minTmpCIMs1 + minCatDiff;
                
            end
            threshold = repmat(mean(tmpTh), bufferNode, 1);
        else
            threshold(1:numNodes) = mean(threshold);
        end
        
        
    elseif sampleNum > bufferNode
        
        % Similarity measure for a numerical attribute
        tmpCIMs1 = CIM(input(:,attType==0), weight(:,attType==0), mean(adaptiveSig));
        
        % Similarity measure for a categorical attribute
        tmpDiff = double( ne(input(:,attType==1), weight(:,attType==1))' ); % A~=B:1, A==B:0
        catDiff = sum(tmpDiff, 1);
        catDiff = catDiff./max(catDiff);
        
        % Multiply a ratio of the number of attributes
        tmpCIMs1 = tmpCIMs1 * size(find(attType==0),2)/size(attType,2);
        catDiff = catDiff * size(find(attType==1),2)/size(attType,2);
        
        % Sum up the similarity by numerical and categorical attributes
        totalSimilarity = tmpCIMs1 + catDiff;
        
        % Set similarity between the local winner nodes and the input for Vigilance Test.
        [Vs1, s1] = min(totalSimilarity);
        totalSimilarity(s1) = inf;
        [Vs2, s2] = min(totalSimilarity);
        
        if threshold(s1) < Vs1 % Case 1 i.e., V < CIM_k1
            % Add Node
            numNodes = numNodes + 1;
            weight(numNodes,:) = input;
            activeNodeIdx = updateActiveNode(activeNodeIdx, numNodes, Lambda);
            CountNode(numNodes) = 1;
            adaptiveSig(numNodes) = SigmaEstimationByNode(weight(:,attType==0), bufferNode, activeNodeIdx);
            CountLabel(numNodes,label) = 1;
            
            % Update a counter for each category on a categorical attribute of s1 weight
            countCategory = [countCategory; InitCountCategory];
            for m = 1:size(input(1,attType==1),2)
                Xcat = input(1,attType==1);
                att = countCategory{numNodes,m};
                att(1,Xcat(m)) = att(1,Xcat(m)) + 1; % Update a count of k-th category
                countCategory{numNodes,m} = att;
            end
            
            % Assigne similarlity threshold
            threshold(numNodes) = threshold(s1);
            
        else % Case 2 i.e., V >= threshold_k1
            
            % Update a counter for each category on a categorical attribute of s1 weight
            for m = 1:size(input(1,attType==1),2)
                Xcat = input(1,attType==1);
                att = countCategory{s1,m};
                att(1,Xcat(m)) = att(1,Xcat(m)) + 1; % Update a count of k-th category
                countCategory{s1,m} = att;
            end
            
            % Update categorical attributes of s1 weight
            idxWcat = find(attType==1);
            for m = 1:size(idxWcat,2) % for each categorical attribute
                
                gradCount = countCategory{s1,m};
                maxval = max(gradCount);
                idxMax = find(gradCount == maxval); % find a category that shows the highest number of counts
                
                if size(idxMax,2) > 1 % if some counters show the maximum value,
                    Xcat = input(1,attType==1);
                    tmpIdx = find(idxMax == Xcat(m), 1);
                    if isempty(tmpIdx) ~= 1 % if the number of counts is the same in the category
                        idxMax = idxMax(1,tmpIdx); % use the latest counted category
                    else
                        idxMax = idxMax(randperm(size(idxMax,2),1)); % random selection from categories that have the same number of counts
                    end
                end
                weight(s1,idxWcat(m)) = idxMax;
            end
            
            CountNode(s1) = CountNode(s1) + 1;
            CountLabel(s1,label) = CountLabel(s1, label) + 1;
            activeNodeIdx = updateActiveNode(activeNodeIdx, s1, Lambda);
            
            % Update numerical attributes of s1 weight
            weight(s1,attType==0) = weight(s1,attType==0) + (1/CountNode(s1)) * (input(:,attType==0) - weight(s1,attType==0));
            
            if threshold(s2) >= Vs2 % Case 3 i.e., V >= threshold_k2
                
                % Update categorical attributes of s2 weight
                idxWcat = find(attType==1);
                for m = 1:size(idxWcat,2)
                    
                    gradCount = countCategory{s2,m};
                    maxval = max(gradCount);
                    idxMax = find(gradCount == maxval); % find a category that shows the highest number of counts
                    
                    if size(idxMax,2) > 1 % if counters have the same value
                        Xcat = input(1,attType==1);
                        tmpIdx = find(idxMax == Xcat(m), 1);
                        if isempty(tmpIdx) ~= 1 % if the number of counts is the same among a category
                            idxMax = idxMax(1,tmpIdx); % use the latest counted category
                        else
                            idxMax = idxMax(randperm(size(idxMax,2),1)); % random selection from categories that have the same number of counts
                        end
                    end
                    weight(s2,idxWcat(m)) = idxMax;
                end
                
                % Update weight of s2 node.
                weight(s2,attType==0) = weight(s2,attType==0) + (1/10*CountNode(s2)) * (input(:,attType==0) - weight(s2,attType==0));
            end
            
        end % if minCIM < Lcim_s1 % Case 1 i.e., V < CIM_k1
    end % if size(weight,1) < 2
    
    
end % for sampleNum = 1:size(DATA,1)



net.numNodes = numNodes;      % Number of nodes
net.weight = weight;          % Mean of nodes
net.CountNode = CountNode;    % Counter for each node
net.adaptiveSig = adaptiveSig;
net.Lambda = Lambda;
net.adaptiveSig = adaptiveSig;
net.activeNodeIdx = activeNodeIdx;
net.threshold = threshold;
net.CountLabel = CountLabel;
net.countCategory = countCategory;

end


% Compute an initial kernel bandwidth for CIM based on data points.
function estSig = SigmaEstimationByNode(weight, bufferNode, activeNodeIdx)

if (size(weight,1) - bufferNode) <= 0
    exNodes = weight;
elseif (size(weight,1) - bufferNode) > 0
    idx = size(activeNodeIdx,2):-1:bufferNode; % Extract new active nodes
    exNodes = weight(activeNodeIdx(idx),:);
end

% Add a small value for handling categorical data.
qStd = std(exNodes);
qStd(qStd==0) = 1.0E-6;

% normal reference rule-of-thumb
% https://www.sciencedirect.com/science/article/abs/pii/S0167715212002921
[n,d] = size(exNodes);
estSig = median( ((4/(2+d))^(1/(4+d))) * qStd * n^(-1/(4+d)) );

end


% Correntropy induced Metric
function cim = CIM(X,Y,sig)
% X : 1 x n
% Y : m x n
cim = sqrt(1 - mean(exp(-(X-Y).^2/(2*sig^2)), 2))';
end


function activeNodeIdx = updateActiveNode(activeNodeIdx, winnerIdx, Lambda)

if Lambda*2 < size(activeNodeIdx,2) % (Lambda*2): Provide a sufficient node index buffer for node deletion
    exceededNode = size(activeNodeIdx,2) - Lambda*2;
    deleteNode = 1:(exceededNode+1);
    activeNodeIdx(deleteNode) = [];
    activeNodeIdx = [activeNodeIdx, winnerIdx];
else
    activeNodeIdx = [activeNodeIdx, winnerIdx];
end
end


