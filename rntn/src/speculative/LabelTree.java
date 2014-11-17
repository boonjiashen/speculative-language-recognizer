package speculative;

import java.util.List;

import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;

public class LabelTree {

    public static void main(String[] args) {
        
    }

    public static void labelTree(CoreMap sentence, List<Integer[]> cues) {
        Tree tree = sentence.get(SentimentCoreAnnotations.AnnotatedTree.class);
        
    }
}
