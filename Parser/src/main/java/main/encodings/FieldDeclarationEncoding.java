package main.encodings;

import com.github.javaparser.ast.body.FieldDeclaration;

import java.util.ArrayList;

public class FieldDeclarationEncoding {
    private String modifier;
    private boolean isStatic;
    private String type;
    private ArrayList<IdentifierEncoding> encodingsOfField;
    private ArrayList<Boolean> isAssingeds;
    private boolean hasComment;
    private int numOfVars;

    public FieldDeclarationEncoding(String modifier, boolean isStatic, String type, ArrayList<IdentifierEncoding> encodingsOfField, ArrayList<Boolean> isAssingeds, boolean hasComment, int numOfVars) {
        this.modifier = modifier;
        this.isStatic = isStatic;
        this.type = type;
        this.encodingsOfField = encodingsOfField;
        this.isAssingeds = isAssingeds;
        this.hasComment = hasComment;
        this.numOfVars = numOfVars;
    }


    public static FieldDeclarationEncoding getEncodingFromFieldDeclaration(FieldDeclaration f) {
        String modifier = getModifier(f);
        boolean isStatic = f.isStatic();
        ArrayList<String> types = new ArrayList<>();
        ArrayList<IdentifierEncoding> encodingsOfField = new ArrayList<>();
        ArrayList<Boolean> isAssingeds = new ArrayList<>();
        f.getVariables().forEach(variableDeclarator -> {
            String type = variableDeclarator.getType().asString();
            types.add(type);
            String name = variableDeclarator.getName().asString();
            IdentifierEncoding encodingOfField = IdentifierEncoding.getEncoding("field", name);
            encodingsOfField.add(encodingOfField);
            boolean isAssigned = variableDeclarator.getInitializer().isPresent();
            isAssingeds.add(isAssigned);
        });
        boolean hasComment = f.getComment().isPresent();
        return new FieldDeclarationEncoding(modifier, isStatic, types.get(0), encodingsOfField, isAssingeds, hasComment, encodingsOfField.size());
    }

    // public protected default private
    public static String getModifier(FieldDeclaration fieldDeclaration) {
        if (fieldDeclaration.isPublic()) {
            return "public";
        } else if (fieldDeclaration.isProtected()) {
            return "protected";
        } else if (fieldDeclaration.isPrivate()) {
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
        sb.append(",\"encodingsOfField\":")
                .append(encodingsOfField);
        sb.append(",\"isAssingeds\":")
                .append(isAssingeds);
        sb.append(",\"hasComment\":")
                .append(hasComment);
        sb.append(",\"numOfVars\":")
                .append(numOfVars);
        sb.append('}');
        return sb.toString();
    }
}
