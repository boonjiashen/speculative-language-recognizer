package speculative;

import java.util.List;

/*
 * Represents a sentence as a string array and a list of cue start/end index pairs
 */
public class Sentence {

    public String[] words = null;
    public List<Integer[]> spec = null;
    public String label = null;
    public String defaultLabel = null;

    public Sentence(String[] words, List<Integer[]> spec, String label, String defaultLabel) {
        this.words = words;
        this.spec = spec;
        this.label = label;
        this.defaultLabel = defaultLabel;
    }
}