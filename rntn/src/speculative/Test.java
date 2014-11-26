package speculative;

import java.io.BufferedReader;
import java.util.Collections;
import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.sentiment.SentimentUtils;
import edu.stanford.nlp.trees.MemoryTreebank;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Generics;

public class Test {

    private static enum Input {
        TEXT, TREES
    }
    private static final String DEFAULT_TLPP_CLASS = "edu.stanford.nlp.parser.lexparser.EnglishTreebankParserParams";

    /*
     * Classifies test input using trained model
     * If inputFormat is TREES (default), also generates accuracy, F1 scores
     */
    public static void main(String[] args) throws Exception {

        String filename = "res/test.txt";
        String sentimentModel = "res/model.ser.gz";
        Input inputFormat = Input.TREES;
        boolean stdin = false;
        boolean evaluate = true;

        for (int argIndex = 0; argIndex < args.length; argIndex++) {
            if (args[argIndex].equalsIgnoreCase("-input")) {
                filename = args[++argIndex];
            } else if (args[argIndex].equalsIgnoreCase("-inputFormat")) {
                inputFormat = Input.valueOf(args[++argIndex].toUpperCase());
            } else if (args[argIndex].equalsIgnoreCase("-sentimentModel")) {
                sentimentModel = args[++argIndex];
            } else if (args[argIndex].equalsIgnoreCase("-stdin")) {
                stdin = true;
            } else if (args[argIndex].equalsIgnoreCase("-evaluate")) {
                evaluate = true;
            } else {
                System.err.println("Unknown argument " + args[argIndex]);
                System.exit(2);
            }
        }

        // We construct two pipelines.  One handles tokenization, if
        // necessary.  The other takes tokenized sentences and converts
        // them to sentiment trees.
        Properties pipelineProps = new Properties();
        Properties tokenizerProps = null;
        pipelineProps.setProperty("sentiment.model", sentimentModel);
        if (stdin) {
            pipelineProps.setProperty("ssplit.eolonly", "true");
        }
        if (inputFormat == Input.TREES) {
            pipelineProps.setProperty("annotators", "binarizer, sentiment");
            pipelineProps.setProperty("customAnnotatorClass.binarizer", "edu.stanford.nlp.pipeline.BinarizerAnnotator");
            pipelineProps.setProperty("binarizer.tlppClass", DEFAULT_TLPP_CLASS);
            pipelineProps.setProperty("enforceRequirements", "false");
        } else {
            pipelineProps.setProperty("annotators", "parse, sentiment");
            pipelineProps.setProperty("enforceRequirements", "false");
            tokenizerProps = new Properties();
            tokenizerProps.setProperty("annotators", "tokenize, ssplit");
        }

        int count = 0;
        if (filename != null) count++;
        if (stdin) count++;
        if (count > 1) {
            throw new IllegalArgumentException("Please only specify one of -file or -stdin");
        }
        if (count == 0) {
            throw new IllegalArgumentException("Please specify either -file or -stdin");
        }
        if (evaluate && !inputFormat.equals(Input.TREES)) {
            throw new IllegalArgumentException("Evaluation requires -inputFormat TREES");
        }

        StanfordCoreNLP tokenizer = (tokenizerProps == null) ? null : new StanfordCoreNLP(tokenizerProps);
        StanfordCoreNLP pipeline = new StanfordCoreNLP(pipelineProps);

        if (filename != null) {
            // Process a file.  The pipeline will do tokenization, which
            // means it will split it into sentences as best as possible
            // with the tokenizer.
            List<Annotation> annotations = getAnnotations(tokenizer, inputFormat, filename, true);
            double truePositives = 0.0;
            double falsePositives = 0.0;
            double trueNegatives = 0.0;
            double falseNegatives = 0.0;
            for (Annotation annotation : annotations) {
                pipeline.annotate(annotation);
                for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
                    System.out.println(sentence);
                    System.out.println("  " + sentence.get(SentimentCoreAnnotations.ClassName.class));
                    if (evaluate) {
                        // Note: this assumes 0 is negative class and 1 is positive class
                        Tree tree = sentence.get(SentimentCoreAnnotations.AnnotatedTree.class);
                        int gold = RNNCoreAnnotations.getGoldClass(tree);
                        int predicted = RNNCoreAnnotations.getPredictedClass(tree);
                        if (gold == 1) {
                            if (predicted == 1) {
                                truePositives++;
                            } else {
                                falseNegatives++;
                            }
                        } else {
                            if (predicted == 1) {
                                falsePositives++;
                            } else {
                                trueNegatives++;
                            }
                        }
                    }
                }
            }
            if (evaluate) {
                double total = truePositives + trueNegatives + falsePositives + falseNegatives;
                double accuracy = (truePositives + trueNegatives) / total;
                double precision = truePositives / (truePositives + falsePositives);
                double recall = truePositives / (truePositives + falseNegatives);
                double f1 = 2 * precision * recall / (precision + recall);
                System.out.println("Number of true positives: " + Math.round(truePositives));
                System.out.println("Number of true negatives: " + Math.round(trueNegatives));
                System.out.println("Number of false positives: " + Math.round(falsePositives));
                System.out.println("Number of false negatives: " + Math.round(falseNegatives));
                System.out.println("Total number of test sentences: " + Math.round(total));
                System.out.println("Accuracy: " + accuracy);
                System.out.println("Precision: " + precision);
                System.out.println("Recall: " + recall);
                System.out.println("F1 score: " + f1);
            }
        } else {
            // Process stdin.  Each line will be treated as a single sentence.
            System.err.println("Reading in text from stdin.");
            System.err.println("Please enter one sentence per line.");
            System.err.println("Processing will end when EOF (Ctrl-D) is reached.");
            BufferedReader reader = new BufferedReader(IOUtils.encodedInputStreamReader(System.in, "utf-8"));
            while (true) {
                String line = reader.readLine();
                if (line == null) {
                    break;
                }
                line = line.trim();
                if (line.length() > 0) {
                    Annotation annotation = tokenizer.process(line);
                    pipeline.annotate(annotation);
                    for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
                        System.out.println("  " + sentence.get(SentimentCoreAnnotations.ClassName.class));
                    }
                } else {
                    // Output blank lines for blank lines so the tool can be
                    // used for line-by-line text processing
                    System.out.println("");
                }
            }

        }
    }

    /*
     * Reads an annotation from the given filename using the requested input.
     */
    public static List<Annotation> getAnnotations(StanfordCoreNLP tokenizer, Input inputFormat, String filename, boolean filterUnknown) {
        switch (inputFormat) {
        case TEXT: {
            String text = IOUtils.slurpFileNoExceptions(filename);
            Annotation annotation = new Annotation(text);
            tokenizer.annotate(annotation);
            List<Annotation> annotations = Generics.newArrayList();
            for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
                Annotation nextAnnotation = new Annotation(sentence.get(CoreAnnotations.TextAnnotation.class));
                nextAnnotation.set(CoreAnnotations.SentencesAnnotation.class, Collections.singletonList(sentence));
                annotations.add(nextAnnotation);
            }
            return annotations;
        }
        case TREES: {
            List<Tree> trees;
            if (filterUnknown) {
                trees = SentimentUtils.readTreesWithGoldLabels(filename);
                trees = SentimentUtils.filterUnknownRoots(trees);
            } else {
                trees = Generics.newArrayList();
                MemoryTreebank treebank = new MemoryTreebank("utf-8");
                treebank.loadPath(filename, null);
                for (Tree tree : treebank) {
                    trees.add(tree);
                }
            }

            List<Annotation> annotations = Generics.newArrayList();
            for (Tree tree : trees) {
                CoreMap sentence = new Annotation(Sentence.listToString(tree.yield()));
                sentence.set(TreeCoreAnnotations.TreeAnnotation.class, tree);
                List<CoreMap> sentences = Collections.singletonList(sentence);
                Annotation annotation = new Annotation("");
                annotation.set(CoreAnnotations.SentencesAnnotation.class, sentences);
                annotations.add(annotation);
            }
            return annotations;
        }
        default:
            throw new IllegalArgumentException("Unknown format " + inputFormat);
        }
    }
}
