function f = CloseAllFigures()
    delete(findall(0,'Type','figure'));
    f = true;
end