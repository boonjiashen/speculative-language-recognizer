function [ words_after_replacement ] = replace_sentence_start_and_end( ...
    words, ...
    start_token, ...
    end_token ...
    )
%REPLACE_SENTENCE_START_AND_END Looks in a window of words for the start
%and end of a sentence and replaces words outside the sentence with the
%appropriate tokens. The fullstop is used to find the start and end of a
%sentence.

% For example, {'Dog', 'ate', '.', 'I', 'saw'} will become {'Dog', 'ate', '.',
% '</s>', '</s>'}

words2 = words;  % cell array to be returned
context_size = (length(words) - 1) / 2;  % no. of words before center word
window_size = length(words);

% Look for fullstops to signify start/end of sentence
is_fullstop = strcmp('.', words);
fullstop_inds = find(is_fullstop);  % indices of fullstop word

% If fullstop left of center word, everything from the fullstop to the
% left is padded with a start pad
if any(fullstop_inds <= context_size)    
    % Get right-most index that's still left of center word
    right_ind = max(fullstop_inds(fullstop_inds <= context_size));

    words2([1:window_size] <= right_ind) = {start_token};
end

% If fullstop is at center word or after, everything to the right of
% the fullstop is padded with a end pad
if any(fullstop_inds > context_size)

    % Get left-most index that's still at or to the right of the center
    % word
    left_ind = min(fullstop_inds(fullstop_inds > context_size));

    words2([1:window_size] > left_ind) = {end_token};
end

words_after_replacement = words2;

end

