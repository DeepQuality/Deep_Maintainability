package main;

import main.combine.MultiThreads;
import main.combine.OneFileParsing;
import main.encodings.*;

public class Main {
    public static void main(String[] args) throws Exception {
        int a=0, b=0;
        String parseType = args[0];
        switch (parseType) {
            case "metric":
                Metrics.parseDir(args[1], args[2]);
                break;
            case "one":
                String s1 = new OneFileParsing(args[1]).parse();
                String s2 = Metrics.parseFile(args[1]);
                System.out.print(s1 + s2);
                break;
            case "multiThread":
                MultiThreads.parseDir(args[1], args[2]);
                break;
            default:
                System.err.println("ARGS PARSING ERROR");
                break;
        }
    }
}
