function clf = get_classifier(model)
% Return classifier handle from the model struct or the model itself.
if isstruct(model) && isfield(model,'Classifier')
    clf = model.Classifier;
else
    clf = model;
end
end
