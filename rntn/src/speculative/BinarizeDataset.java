package speculative;

import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.parser.lexparser.TreeBinarizer;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.sentiment.CollapseUnaryTransformer;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.Trees;

public class BinarizeDataset {

    /*
     * Turns text input into trees for use in a RNTN classifier such as
     * the treebank used in the Sentiment project.
     */
    public static List<String> binarizeAndLabel(List<Sentence> sentences) {
        List<String> output = new ArrayList<String>();
        CollapseUnaryTransformer transformer = new CollapseUnaryTransformer();

        String parserModel = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz";

        LexicalizedParser parser = LexicalizedParser.loadModel(parserModel);
        TreeBinarizer binarizer = TreeBinarizer.simpleTreeBinarizer(parser.getTLPParams().headFinder(), parser.treebankLanguagePack());

        System.out.println("Generating parse trees...");
        for (int i = 0; i < sentences.size(); i++) {
            System.out.println("Now binarizing and labeling sentence " + i);
            Sentence s = sentences.get(i);
            List<HasWord> tokens = s.words;

            // Parse the sentence to generate binary tree
            Tree tree = parser.apply(tokens);
            Tree binarized = binarizer.transformTree(tree);
            Tree collapsedUnary = transformer.transformTree(binarized);

            // Processing to annotate each node of tree with indices
            // indicating the words spanned by the node
            Trees.convertToCoreLabels(collapsedUnary);
            collapsedUnary.indexSpans();

            // Label the tree
            labelTree(collapsedUnary, s);
            // Add to output
            output.add(collapsedUnary.toString());
        }
        System.out.println("Done generating parse trees");
        return output;
    }

    /*
     * Labels the tree recursively
     * Any node that spans ALL the words of a cue phrase
     * is labeled speculative (1)
     * All other nodes are labeled non-speculative (0)
     */
    private static void labelTree(Tree tree, Sentence s) {
        if (tree.isLeaf())
            return;

        for (Tree child : tree.children()) {
            labelTree(child, s);
        }

        if (!(tree.label() instanceof CoreLabel)) {
            throw new AssertionError("Expected CoreLabels");
        }
        CoreLabel label = (CoreLabel) tree.label();
        // Label the node non-speculative (0) by default
        label.setValue(s.defaultLabel);
        // Do not process further if the sentence is non-speculative
        if (s.spec.isEmpty()) {
            return;
        }
        // Check the words spanned by the node
        // Label node speculative (1) if the cue phrase is included
        int start = label.get(CoreAnnotations.BeginIndexAnnotation.class);
        int end = label.get(CoreAnnotations.EndIndexAnnotation.class);
        for (Integer[] pair : s.spec) {
            if (pair[0] >= start && pair[1] < end) {
                label.setValue(s.label);
                return;
            }
        }
    }
}
