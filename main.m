rng(1267);

opt.d = 50;
opt.eta0 = 5e-3;
opt.maxIter = 200;
opt.tied = true;
opt.gradient_check = false;
opt.momentum = 0.5;
opt.eco = false;
%opt.nonlinear_h = 'tanh';
opt.nonlinear_h = 'identity';
opt.nonlinear_g = 'identity';
opt.dataset = 'blog';
opt.output = '/tmp';
opt.train_size = 0.1;
opt.normalize_laplacian = true;
opt.algorithm = 'gae';

[G,Y] = DataLoader(opt.dataset);

% compute the laplacian
n = size(G,1);
L = spdiags(sum(G,2),0,n,n)-G;
if opt.normalize_laplacian
    D_inv_sqrt = spdiags(1./sqrt(sum(G,2)),0,n,n);
    L = D_inv_sqrt*L*D_inv_sqrt;
end

switch opt.algorithm
    case 'spectral' % spectral graph basis
        [U,~] = eigs(L, opt.d, 'SM');
    case 'gae' % graph autoencoder
        [U,~] = GraphAutoencoder(G, opt);
    otherwise
        error('algorithm %s is not defined.\n', opt.algorithm);
end

%% save model parameters
if ~exist(opt.output)
    mkdir(opt.output);
end
save(strcat([opt.output,'/params.mat']), 'U');

% normalization as suggested by Ng et al.
U = bsxfun(@rdivide, U, sqrt(sum(U.^2,2)));

% randomly partition the data
shuffled_index = randperm(n);
train_index = shuffled_index(1:floor(n*opt.train_size));
test_index = shuffled_index(floor(n*opt.train_size)+1:end);

stat_info = LibSVMClassify(U(train_index,:), Y(train_index,:), ...
    U(test_index,:), Y(test_index,:));
fprintf('macro_F1=%f micro_F1=%f\n', stat_info(4), stat_info(8));

%%% rearrange the graph using spectral clustering
%function G = rearrange_nodes(G, k)
    %Y = SpectralClustering(G, k, 3);
    %[~,y] = max(Y,[],2);
    %[~,pi] = sort(y);
    %G = G(pi,:);
    %G = G(:,pi)
%end
