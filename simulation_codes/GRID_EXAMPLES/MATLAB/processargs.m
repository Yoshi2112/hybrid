function processbatch(var1,var2,outfilename)


    fprintf('Processing with inputs %s %s\n',var1,var2);

    outval = var1 + var2;

    save(outfilename,'outval');

    fprintf('Saved image as %s\n',outfilename);

end
