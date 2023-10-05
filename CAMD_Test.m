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
function [NMI, ARI] = CAMD_Test(DATA, LABELS, net, attType)

weight = net.weight;
adaptiveSig = net.adaptiveSig;

if size(weight,1) <= 1
    NMI = nan;
    ARI = nan;
else
    
    % Assume each node has different label information
    LabelCluster = 1:1:size(weight,1);
    
    % Classify test data by disjoint clusters
    EstLabel = zeros(size(LABELS));
    for sampleNum = 1:size(DATA,1)
        
        % Current data sample
        pattern = DATA(sampleNum,:); % Current Input
        
        % Find 1st winner node
        % Similarity measure for a numerical attribute
        tmpCIMs1 = CIM(pattern(:,attType==0), weight(:,attType==0), mean(adaptiveSig));
        
        % Similarity measure for a categorical attribute
        tmpDiff = double( ne(pattern(:,attType==1), weight(:,attType==1))' ); % A~=B:1, A~=B:0
        catDiff = sum(tmpDiff, 1);
        catDiff = catDiff./max(catDiff);
        
        % Multiply a ratio of the number of attributes
        tmpCIMs1 = tmpCIMs1 * size(find(attType==0),2)/size(attType,2);
        catDiff = catDiff * size(find(attType==1),2)/size(attType,2);
        
        % Sum up the similarity by numerical and categorical attributes
        totalSimilarity = tmpCIMs1 + catDiff;
        [~, s1] = min(totalSimilarity);
        
        EstLabel(sampleNum, 1) = LabelCluster(1, s1);
    end
    
    
    % Compute Mutual Information
    [NMI, ~] = NormalizedMutualInformation( LABELS, EstLabel );
    
    % Compute Adjusted Rand Index
    ARI = AdjustedRandIndex( LABELS, EstLabel );
    
end

end


% Correntropy induced Metric (Gaussian Kernel based)
function cim = CIM(X,Y,sig)
% X : 1 x n
% Y : m x n
cim = sqrt(1 - mean(exp(-(X-Y).^2/(2*sig^2)), 2))';
end

function [normMI, MI] = NormalizedMutualInformation(x, y)
% Compute normalized mutual information I(x,y)/sqrt(H(x)*H(y)) of two discrete variables x and y.
% Input:
%   x, y: two integer vector of the same length
% Ouput:
%   normMI: normalized mutual information normMI=I(x,y)/sqrt(H(x)*H(y))
% Written by Mo Chen (sth4nth@gmail.com).
assert(numel(x) == numel(y));
n = numel(x);
x = reshape(x,1,n);
y = reshape(y,1,n);

l = min(min(x),min(y));
x = x-l+1;
y = y-l+1;
k = max(max(x),max(y));

idx = 1:n;
Mx = sparse(idx,x,1,n,k,n);
My = sparse(idx,y,1,n,k,n);
Pxy = nonzeros(Mx'*My/n); %joint distribution of x and y
Hxy = -dot(Pxy,log2(Pxy));


% hacking, to elimative the 0log0 issue
Px = nonzeros(mean(Mx,1));
Py = nonzeros(mean(My,1));

% entropy of Py and Px
Hx = -dot(Px,log2(Px));
Hy = -dot(Py,log2(Py));

% mutual information
MI = Hx + Hy - Hxy;

% normalized mutual information
z = sqrt((MI/Hx)*(MI/Hy));
normMI = max(0,z);

end


function ARI = AdjustedRandIndex(ACTUAL, PREDICTED)

%function adjrand=adjrand(u,v)
%
% Computes the adjusted Rand index to assess the quality of a clustering.
% Perfectly random clustering returns the minimum score of 0, perfect
% clustering returns the maximum score of 1.
%
%INPUTS
% u = the labeling as predicted by a clustering algorithm
% v = the true labeling
%
%OUTPUTS
% adjrand = the adjusted Rand index
%
%
%Author: Tijl De Bie, february 2003.

n=length(PREDICTED);
ku=max(PREDICTED);
kv=max(ACTUAL);
m=zeros(ku,kv);
for i=1:n
    m(PREDICTED(i),ACTUAL(i))=m(PREDICTED(i),ACTUAL(i))+1;
end
mu=sum(m,2);
mv=sum(m,1);

a=0;
for i=1:ku
    for j=1:kv
        if m(i,j)>1
            a=a+nchoosek(m(i,j),2);
        end
    end
end

b1=0;
b2=0;
for i=1:ku
    if mu(i)>1
        b1=b1+nchoosek(mu(i),2);
    end
end
for i=1:kv
    if mv(i)>1
        b2=b2+nchoosek(mv(i),2);
    end
end

c=nchoosek(n,2);

ARI=(a-b1*b2/c)/(0.5*(b1+b2)-b1*b2/c);

if ARI<0
    ARI = 0;
end


end

