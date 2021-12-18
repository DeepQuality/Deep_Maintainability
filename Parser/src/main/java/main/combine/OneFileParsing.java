package main.combine;

import ast.parser.AST;
import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseProblemException;
import com.github.javaparser.ParseResult;
import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.SimpleName;
import com.github.javaparser.printer.DotPrinter;
import com.github.javaparser.printer.lexicalpreservation.LexicalPreservingPrinter;
import main.encodings.FieldDeclarationEncoding;
import main.encodings.IdentifierEncoding;
import main.encodings.MethodDeclarationEncoding;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;

public class OneFileParsing {
    private String path;
    public OneFileParsing(String path) {
        this.path = path;
    }

    public String parse() {
        File sourceFile = new File(this.path);
        try {
            CompilationUnit compilationUnit = getCompilationUnit(sourceFile);
            assert compilationUnit != null;
            ArrayList<ClassOrInterfaceDeclaration> targets = getClassOrInterfaceDeclarations(compilationUnit);
            if (targets.size() != 1) {
                StringBuilder result = new StringBuilder();
                result.append("EXCEPTION-----MetaData-----\n")
                        .append(sourceFile.getAbsolutePath())
                        .append('\n')
                        .append("EXCEPTION:Number of class, ")
                        .append(targets.size());
                return result.toString();
            } else {
                ClassOrInterfaceDeclaration classDeclaration = targets.get(0);
                StringBuilder result = new StringBuilder();
                result.append("-----MetaData-----\n")
                        .append(sourceFile.getAbsolutePath()+"\t"+classDeclaration.getNameAsString())
                        .append('\n')
                        .append("-----AST-----\n")
                        .append(getAstStr(classDeclaration))
                        .append('\n')
                        .append("-----Code Sequence With Abstract-----\n")
                        .append(getCodeWithAbs(classDeclaration))
                        .append('\n')
                        .append("-----Field Declarations-----\n")
                        .append(getFieldDeclarationJson(classDeclaration))
                        .append('\n')
                        .append("-----Method Declarations-----\n")
                        .append(getMethodDeclarationJson(classDeclaration))
                        .append('\n')
                        .append("-----Identifiers-----\n")
                        .append(getAllIdensJson(targets))
                        .append('\n');
                return result.toString();
            }
        } catch (ParseProblemException ignored) {
            StringBuilder result = new StringBuilder();
            result.append("EXCEPTION-----MetaData-----\n")
                    .append("EXCEPTION:ParseProblemException")
                    .append("\n");
            return result.toString();
        } catch (Exception e) {
            StringBuilder result = new StringBuilder();
            e.printStackTrace();
            result.append("EXCEPTION-----MetaData-----\n")
                    .append("EXCEPTION:" + e.getClass().getName())
                    .append("\n");
            return result.toString();
        }
    }

    public static CompilationUnit getCompilationUnit(File sourceFile) throws FileNotFoundException {
        CompilationUnit compilationUnit;
        ParserConfiguration configuration = new ParserConfiguration();
        ParseResult<CompilationUnit> parseResult = new JavaParser(configuration).parse(sourceFile);
        if (parseResult.isSuccessful()) {
            compilationUnit = parseResult.getResult().orElse(null);
        } else {
            throw new ParseProblemException(parseResult.getProblems());
        }
        return compilationUnit;
    }

    private String getAllIdensJson(ArrayList<ClassOrInterfaceDeclaration> targets) {
        ArrayList<IdentifierEncoding> allIdens = new ArrayList<>();
        targets.get(0).findAll(SimpleName.class).forEach(simpleName -> allIdens.add(IdentifierEncoding.getEncoding("Identifier", simpleName.asString())));
        StringBuilder allIdensJson = new StringBuilder("{");
        allIdensJson.append("\"Identifiers\":")
                .append(allIdens);
        allIdensJson.append('}');
        return allIdensJson.toString().trim();
    }

    private String getMethodDeclarationJson(ClassOrInterfaceDeclaration classDeclaration) {
        ArrayList<MethodDeclarationEncoding> methodDeclarations = new ArrayList<>();
        classDeclaration.findAll(MethodDeclaration.class)
                .forEach(methodDeclaration -> {
                            MethodDeclarationEncoding methodDeclarationEncoding = MethodDeclarationEncoding.getEncodingFromMethodDeclaration(methodDeclaration);
                            methodDeclarations.add(methodDeclarationEncoding);
                        }
                );
        StringBuilder methodDeclartionsJson = new StringBuilder("{");
        methodDeclartionsJson.append("\"MethodDeclarations\":")
                .append(methodDeclarations);
        methodDeclartionsJson.append('}');
        return methodDeclartionsJson.toString().trim();
    }

    private String getFieldDeclarationJson(ClassOrInterfaceDeclaration classDeclaration) {
        ArrayList<FieldDeclarationEncoding> fieldDeclarations = new ArrayList<>();
        classDeclaration.findAll(FieldDeclaration.class)
                .forEach(fieldDeclaration -> {
                            FieldDeclarationEncoding fieldDeclarationEncoding = FieldDeclarationEncoding.getEncodingFromFieldDeclaration(fieldDeclaration);
                            fieldDeclarations.add(fieldDeclarationEncoding);
                        }
                );
        StringBuilder fieldDeclartionsJson = new StringBuilder("{");
        fieldDeclartionsJson.append("\"FieldDeclarations\":")
                .append(fieldDeclarations);
        fieldDeclartionsJson.append('}');
        return fieldDeclartionsJson.toString().trim();
    }

    private String getCodeWithAbs(ClassOrInterfaceDeclaration classDeclaration) {
        ClassOrInterfaceDeclaration cu = LexicalPreservingPrinter.setup(classDeclaration);
        String codeWithAbs = LexicalPreservingPrinter.print(cu);
        return codeWithAbs.trim();
    }

    private String getAstStr(ClassOrInterfaceDeclaration classOrInterfaceDeclaration) {
        DotPrinter printer = new DotPrinter(true);
        ArrayList<String> asts = new ArrayList<>();
        classOrInterfaceDeclaration.findAll(MethodDeclaration.class)
                .forEach(methodDeclaration -> {
                            String dot = printer.output(methodDeclaration);
                            AST ast = AST.parseDot(dot);
                            asts.add(ast.toString());
                        }
                );
        StringBuilder astStr = new StringBuilder();
        for (String ast :
                asts) {
            astStr.append(ast.trim());
            astStr.append("\n");
        }
        return astStr.toString().trim();
    }

    public static ArrayList<ClassOrInterfaceDeclaration> getClassOrInterfaceDeclarations(CompilationUnit compilationUnit) {
        List<ClassOrInterfaceDeclaration> classOrInterfaceDeclarations = compilationUnit.findAll(ClassOrInterfaceDeclaration.class);
        ArrayList<ClassOrInterfaceDeclaration> targets = new ArrayList<>();
        for (ClassOrInterfaceDeclaration classOrInterfaceDeclaration : classOrInterfaceDeclarations) {
            if ((!classOrInterfaceDeclaration.isInnerClass()) &&
                    (!classOrInterfaceDeclaration.isInterface() &&
                            classOrInterfaceDeclaration.isPublic())
            ) {
                targets.add(classOrInterfaceDeclaration);
            }
        }
        return targets;
    }

    public static void main(String[] args) {
        String s = new OneFileParsing("E:/temp.java").parse();
        System.out.println(s);
    }
}
