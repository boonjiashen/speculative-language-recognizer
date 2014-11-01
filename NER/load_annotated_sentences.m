function [ labeled_data ] = load_annotated_sentences(filename)
%LOAD_ANNOTATED_SENTENCES Loads sentences into a cell array of
%words/punctuation and labels.
%For each line in filename, we expect a word or punctuation, followed by
%PERSON if the word/punctuation is part of a PERSON named entity, and 0
%otherwise.
%Returns a mx2 cell array where each row is a word/punctuation followed by
%a 0/1 label (1 indicates person)


% Load training data as strings
fid = fopen(filename);

% first cell is itself a cell array of words, second cell is cell array of
% labels
labeled_data = textscan(fid, '%s %s\n');

labeled_data = horzcat(labeled_data{:});  % contacenate to mx2 cell array
fclose(fid);

% Convert labels from strings '0'/'PERSON' to integers 0/1
for wi = 1: size(labeled_data, 1)
    label = labeled_data{wi, 2};
    isPerson = strcmp(label, 'PERSON') == 1;
    if isPerson
        labeled_data{wi, 2} = 1;
    else
        labeled_data{wi, 2} = 0;
    end
end

end

