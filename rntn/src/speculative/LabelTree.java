package speculative;

import java.util.List;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.Label;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;

public class LabelTree {

    public static void main(String[] args) {
        
    }

    public static void labelTree(Tree tree, List<Integer[]> cues) {
        setSpeculationLabels(tree);
    }

    private static void setSpeculationLabels(Tree tree) {
        if (tree.isLeaf()) {
            return;
        }

        for (Tree child : tree.children()) {
            setSpeculationLabels(child);
        }

        Label label = tree.label();
        if (!(label instanceof CoreLabel)) {
            throw new IllegalArgumentException("Required a tree with CoreLabels");
        }
        CoreLabel cl = (CoreLabel) label;
        cl.setValue(Integer.toString(RNNCoreAnnotations.getPredictedClass(tree)));
    }
}
