package com.github.mauricioaniche.ck.util;

import com.github.mauricioaniche.ck.CKClassResult;
import com.github.mauricioaniche.ck.CKMethodResult;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Map;

/**
 * modifide by hym
 */
public class ResultWriter {

    private static final String[] CLASS_HEADER = { "file", "class", "type", "cbo", "wmc", "rfc", "lcom",
            "totalMethodsQty", "staticMethodsQty", "publicMethodsQty", "privateMethodsQty", "protectedMethodsQty", "defaultMethodsQty",
            "visibleMethodsQty", "abstractMethodsQty", "finalMethodsQty", "synchronizedMethodsQty", "totalFieldsQty", "staticFieldsQty", "publicFieldsQty",
            "privateFieldsQty", "protectedFieldsQty", "defaultFieldsQty", "finalFieldsQty", "synchronizedFieldsQty", "loc",
            "returnQty", "loopQty", "comparisonsQty", "tryCatchQty", "parenthesizedExpsQty", "stringLiteralsQty",
            "numbersQty", "assignmentsQty", "mathOperationsQty", "variablesQty", "maxNestedBlocksQty",
            "anonymousClassesQty", "innerClassesQty", "lambdasQty", "uniqueWordsQty", "logStatementsQty" };
    private static final String[] METHOD_HEADER = { "file", "class", "method", "constructor", "line", "cbo", "wmc", "rfc", "loc",
            "returnsQty", "variablesQty", "parametersQty", "methodsInvokedQty", "methodsInvokedLocalQty", "methodsInvokedIndirectLocalQty", "loopQty", "comparisonsQty", "tryCatchQty",
            "parenthesizedExpsQty", "stringLiteralsQty", "numbersQty", "assignmentsQty", "mathOperationsQty",
            "maxNestedBlocks", "anonymousClassesQty", "innerClassesQty", "lambdasQty", "uniqueWordsQty", "modifiers", "logStatementsQty", "hasJavaDoc" };
    private static final String[] VAR_FIELD_HEADER = { "file", "class", "method", "variable", "usage" };
    private final boolean variablesAndFields;

    private CSVPrinter classPrinter;

    /**
     * Initialise a new ResultWriter that writes to the specified files. Begins by
     * writing CSV headers to each file.
     * 
     * @param classFile    Output file for class metrics
     * @param methodFile   Output file for method metrics
     * @param variableFile Output file for variable metrics
     * @param fieldFile    Output file for field metrics
     * @throws IOException If headers cannot be written
     */
    public ResultWriter(String classFile, String methodFile, String variableFile, String fieldFile, boolean variablesAndFields) throws IOException {
        FileWriter classOut = new FileWriter(classFile);
        this.classPrinter = new CSVPrinter(classOut, CSVFormat.DEFAULT.withHeader(CLASS_HEADER));
        this.variablesAndFields = variablesAndFields;
    }

    /**
     * Print results for a single class and its methods and fields to the
     * appropriate CSVPrinters.
     * 
     * @param result The CKClassResult
     * @throws IOException If output files cannot be written to
     */
    public void printResult(CKClassResult result) throws IOException {
        this.classPrinter.printRecord(result.getFile(), result.getClassName(), result.getType(), result.getCbo(),
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
                result.getLambdasQty(), result.getUniqueWordsQty(), result.getNumberOfLogStatements());
    }

    /**
     * Flush and close resources that were opened to write results. This method
     * should be called after all CKClassResults have been calculated and printed.
     * 
     * @throws IOException If the resources cannot be closed
     */
    public void flushAndClose() throws IOException {
        this.classPrinter.flush();
        this.classPrinter.close();
    }
}
