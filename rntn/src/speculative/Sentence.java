package speculative;

import java.util.List;

import edu.stanford.nlp.ling.HasWord;

/*
 * Represents a sentence as a string array and a list of cue start/end index pairs
 */
public class Sentence {

    public List<HasWord> words = null;
    public List<Integer[]> spec = null;
    public String label = null;
    public String defaultLabel = null;

    public Sentence(List<HasWord> words, List<Integer[]> spec, String label, String defaultLabel) {
        this.words = words;
        this.spec = spec;
        this.label = label;
        this.defaultLabel = defaultLabel;
    }
}