package speculative;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.sentiment.SentimentPipeline;
//import edu.stanford.nlp.sentiment.SentimentPipeline.Output;
import edu.stanford.nlp.util.CoreMap;
public class Train {

    public static void main(String[] args){

        String line = "The machine might not be working";

        Properties tokenizerProps = null;
        Properties pipelineProps = new Properties();
        tokenizerProps = new Properties();
        tokenizerProps.setProperty("annotators", "tokenize, ssplit");

        pipelineProps.setProperty("annotators", "parse, sentiment");
        pipelineProps.setProperty("enforceRequirements", "false");
        StanfordCoreNLP tokenizer = (tokenizerProps == null) ? null : new StanfordCoreNLP(tokenizerProps);
        StanfordCoreNLP pipeline = new StanfordCoreNLP(pipelineProps);
        // List<SentimentPipeline.Output> outputFormats = Arrays.asList(new Output[] { Output.ROOT });

        if (line.length() > 0) {
            Annotation annotation = tokenizer.process(line);
            pipeline.annotate(annotation);
            for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
                System.out.println(sentence);
                //SentimentPipeline.outputTree(System.out, sentence,outputFormats);
            }
        } else {
            // Output blank lines for blank lines so the tool can be
            // used for line-by-line text processing
            System.out.println("");
        }
    }
}
