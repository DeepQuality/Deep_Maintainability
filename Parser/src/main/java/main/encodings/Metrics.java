package main.encodings;

import com.github.mauricioaniche.ck.CK;
import com.github.mauricioaniche.ck.MetricsExecutor;
import com.github.mauricioaniche.ck.metric.ClassLevelMetric;
import com.github.mauricioaniche.ck.metric.MethodLevelMetric;
import com.github.mauricioaniche.ck.util.MetricsFinder;
import com.github.mauricioaniche.ck.util.ResultWriter;
import org.eclipse.jdt.core.JavaCore;
import org.eclipse.jdt.core.dom.AST;
import org.eclipse.jdt.core.dom.ASTParser;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;

public class Metrics {
    public static void parseDir(String sourceDirName, String prefixDir) throws IOException {
        int maxAtOnce = 0;
        ResultWriter writer = new ResultWriter(prefixDir + "class.csv",
                prefixDir + "method.csv",
                prefixDir + "variable.csv",
                prefixDir + "field.csv",
                true);

        new CK(false, maxAtOnce, true).calculate(sourceDirName, result -> {
            try {
                writer.printResult(result);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
        writer.flushAndClose();
    }


    public static String parseFile(String path) {
        MetricsFinder finder = new MetricsFinder();
        Callable<List<ClassLevelMetric>> classLevelMetrics = finder::allClassLevelMetrics;
        Callable<List<MethodLevelMetric>> methodLevelMetrics = () -> finder.allMethodLevelMetrics(true);
        var ref = new Object() {
            int[] res;
        };
        MetricsExecutor storage = new MetricsExecutor(classLevelMetrics, methodLevelMetrics, result -> {
            ref.res = new int[]{result.getCbo(),
                    result.getWmc(), result.getRfc(), result.getLcom(), result.getNumberOfMethods(),
                    result.getNumberOfStaticMethods(), result.getNumberOfPublicMethods(),
                    result.getNumberOfPrivateMethods(), result.getNumberOfProtectedMethods(),
                    result.getNumberOfDefaultMethods(), result.getVisibleMethods().size(), result.getNumberOfAbstractMethods(),
                    result.getNumberOfFinalMethods(), result.getNumberOfSynchronizedMethods(), result.getNumberOfFields(),
                    result.getNumberOfStaticFields(), result.getNumberOfPublicFields(), result.getNumberOfPrivateFields(),
                    result.getNumberOfProtectedFields(), result.getNumberOfDefaultFields(), result.getNumberOfFinalFields(),
                    result.getNumberOfSynchronizedFields(), result.getLoc(), result.getReturnQty(),
                    result.getLoopQty(), result.getComparisonsQty(), result.getTryCatchQty(),
                    result.getParenthesizedExpsQty(), result.getStringLiteralsQty(), result.getNumbersQty(),
                    result.getAssignmentsQty(), result.getMathOperationsQty(), result.getVariablesQty(),
                    result.getMaxNestedBlocks(), result.getAnonymousClassesQty(), result.getInnerClassesQty(),
                    result.getLambdasQty(), result.getUniqueWordsQty(), result.getNumberOfLogStatements()};
        });

        ASTParser parser = ASTParser.newParser(AST.JLS11);
        parser.setResolveBindings(false);
        parser.setBindingsRecovery(false);

        Map<String, String> options = JavaCore.getOptions();
        JavaCore.setComplianceOptions(JavaCore.VERSION_11, options);
        parser.setCompilerOptions(options);
        parser.setEnvironment(null, null, null, true);
        parser.createASTs(new String[]{path}, null, new String[0], storage, null);
        String s = Arrays.toString(ref.res);
        return s.substring(1, s.length()-1).replace(" ", "");
    }

}
