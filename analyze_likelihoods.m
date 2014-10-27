% Analyze the most indicative words
% Assumes that we have a variable pos_loglikes and neg_loglikes

% Get log-likelihood ratio of all words
dictionary = pos_loglikes.keys;
dict_sz = length(dictionary);
loglikeratios = zeros(dict_sz, 1);
for wi = 1:dict_sz
    word = dictionary{wi};
    loglikeratio = pos_loglikes(word) - neg_loglikes(word);
    loglikeratios(wi) = loglikeratio;
end

%% Print top-N most indicative words

[sorted, inds] = sort(loglikeratios, 'descend');
chart_sz = 20;  % Top-5 most indicative words
for wi = 1:chart_sz
    word = dictionary{inds(wi)};
    fprintf('Most indicative word #%i: %s\n', wi, word);
end

% Print least indicative words
for wi = 1:chart_sz
    word = dictionary{inds(end - wi + 1)};
    fprintf('Least indicative word #%i: %s\n', wi, word);
end
