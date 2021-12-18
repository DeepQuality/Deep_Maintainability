package main.encodings;

import com.github.javaparser.ast.body.MethodDeclaration;
import main.util.Util;

import java.util.ArrayList;

public class MethodDeclarationEncoding {
    private String modifier;
    private boolean isStatic;
    private String type;
    private IdentifierEncoding encodingOfMethodName;
    private ArrayList<String> parameterTypes;
    private ArrayList<IdentifierEncoding> encodingsOfParameterName;
    private boolean hasComment;

    public MethodDeclarationEncoding(String modifier, boolean isStatic, String type, IdentifierEncoding encodingOfMethodName, ArrayList<String> parameterTypes, ArrayList<IdentifierEncoding> encodingsOfParameterName, boolean hasComment) {
        this.modifier = modifier;
        this.isStatic = isStatic;
        this.type = type;
        this.encodingOfMethodName = encodingOfMethodName;
        this.parameterTypes = parameterTypes;
        this.encodingsOfParameterName = encodingsOfParameterName;
        this.hasComment = hasComment;
    }


    public static MethodDeclarationEncoding getEncodingFromMethodDeclaration(MethodDeclaration methodDeclaration) {
        String modifier = getModifier(methodDeclaration);
        boolean isStatic = methodDeclaration.isStatic();
        String type = methodDeclaration.getType().asString();
        IdentifierEncoding encodingsOfMethodName = IdentifierEncoding.getEncoding("method", methodDeclaration.getName().asString());

        ArrayList<String> parameterTypes = new ArrayList<>();
        ArrayList<IdentifierEncoding> encodingsOfParameterName = new ArrayList<>();
        methodDeclaration.getParameters().forEach(parameter -> {
            String parameterType = parameter.getType().asString();
            String parameterName = parameter.getName().asString();
            IdentifierEncoding encodingOfParameterName = IdentifierEncoding.getEncoding("parameter", parameterName);
            parameterTypes.add(parameterType);
            encodingsOfParameterName.add(encodingOfParameterName);
        });
        boolean hasComment = methodDeclaration.getComment().isPresent();
        return new MethodDeclarationEncoding(modifier, isStatic, type, encodingsOfMethodName, parameterTypes, encodingsOfParameterName, hasComment);
    }

    // public protected default private
    public static String getModifier(MethodDeclaration methodDeclaration) {
        if (methodDeclaration.isPublic()) {
            return "public";
        } else if (methodDeclaration.isProtected()) {
            return "protected";
        } else if (methodDeclaration.isPrivate()) {
            return "private";
        } else {
            return "default";
        }
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("{");
        sb.append("\"modifier\":\"")
                .append(modifier).append('\"');
        sb.append(",\"isStatic\":")
                .append(isStatic);
        sb.append(",\"type\":\"")
                .append(type).append('\"');
        sb.append(",\"encodingOfMethodName\":")
                .append(encodingOfMethodName);
        sb.append(",\"parameterTypes\":")
                .append(Util.arrayListStringToString(parameterTypes));
        sb.append(",\"encodingsOfParameterName\":")
                .append(encodingsOfParameterName);
        sb.append(",\"hasComment\":")
                .append(hasComment);
        sb.append('}');
        return sb.toString();
    }
}
