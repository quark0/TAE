function [G,Y] = DataLoader(dataset)

    switch dataset

        case 'facebook'
            % ego-Facebook
            edges = load('data/facebook_combined.txt') + 1;
            n = max(edges(:));
            G = sparse(edges(:,1),edges(:,2),1,n,n);

        case 'blog'
            % BlogCatalog3
            nodes = load('data/BlogCatalog-dataset/data/nodes.csv');
            edges = load('data/BlogCatalog-dataset/data/edges.csv');
            group_edges = load('data/BlogCatalog-dataset/data/group-edges.csv');
            num_groups = 39;
            n = size(nodes,1);
            G = sparse(edges(:,1),edges(:,2),1,n,n);
            Y = sparse(group_edges(:,1),group_edges(:,2),1,n,num_groups);

    end

    G = max(G,G'); % symmetrization
end
