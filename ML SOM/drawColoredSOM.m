function fig = drawColoredSOM(resulted, labels, rows, cols, colors, topology, titl)
    l=rows;
    b=cols;
    C = colors;
    
    if topology == 1
        xhex=[0 1 2 2 1 0]; % x-coordinates of the vertices
        yhex=[2 3 2 1 0 1]; % y-coordinates of the vertices

        h = zeros(length(labels),3); 
        fig = figure;
        axis off
        for i=1:b
            for k=1:l
                m=k-1;
                cls = resulted(l-k+1,b-i+1);
                colorInd = cls;
                if cls~=0
                    h(cls,:) = C(colorInd+1, :);
                end
                patch((xhex+mod(k,2))-2*i,yhex+2*m,C(colorInd+1, :)); % make a hexagon at [2i,2j]
                hold on
            end

        end
        f = zeros(length(h),1);
        for i=1:length(h)
            f(i) = plot(NaN,NaN,'o','MarkerFaceColor',h(i,:), 'MarkerEdgeColor', h(i,:));

        end
        title(titl);
        legend(f,labels);
        axis equal
        ext = '.jpg';
        path = 'SOMWithCOlors/';
        name = titl;
        name = [path, titl, ext];
        saveas(fig, name);
    end
end