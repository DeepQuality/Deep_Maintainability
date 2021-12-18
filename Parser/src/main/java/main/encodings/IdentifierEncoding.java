package main.encodings;

import main.util.Dic;
import main.util.IdentifierSpliter;
import main.util.Util;

import java.util.ArrayList;

public class IdentifierEncoding {
    // Class Field Method Parameter Variable
    private String name;
    private String type;
    private int numOfParts;
    // 1 all lowercase  2 first letter capital  3 all capital  4 other
    private ArrayList<String> partNames;
    private ArrayList<String> partTypes;
    private ArrayList<Boolean> isInDics;
    private boolean containNum;
    private boolean containUnderscore;
    private boolean containDollar;

    public IdentifierEncoding(String name, String type, int numOfParts, ArrayList<String> partNames, ArrayList<String> partTypes, ArrayList<Boolean> isInDics, boolean containNum, boolean containUnderscore, boolean containDollar) {
        this.name = name;
        this.type = type;
        this.numOfParts = numOfParts;
        this.partNames = partNames;
        this.partTypes = partTypes;
        this.isInDics = isInDics;
        this.containNum = containNum;
        this.containUnderscore = containUnderscore;
        this.containDollar = containDollar;
    }

    public static boolean isType1(String part) {
        char[] chars = part.toCharArray();
        for (char aChar : chars) {
            if (!(aChar >= 'a' && aChar <= 'z')) {
                return false;
            }
        }
        return true;
    }

    public static boolean isType2(String part) {
        if (part.length() == 1) {
            return false;
        }
        char[] chars = part.toCharArray();
        if (!(chars[0] >= 'A' && chars[0] <= 'Z')) {
            return false;
        }
        for (int i = 1; i < chars.length; i++) {
            if (!(chars[i] >= 'a' && chars[i] <= 'z')) {
                return false;
            }
        }
        return true;
    }

    public static boolean isType3(String part) {
        char[] chars = part.toCharArray();
        for (char aChar : chars) {
            if (!(aChar >= 'A' && aChar <= 'Z')) {
                return false;
            }
        }
        return true;
    }

    public static String getPartType(String part) {
        if (isType1(part)) {
            return "1";
        } else if (isType2(part)) {
            return "2";
        } else if (isType3(part)) {
            return "3";
        } else {
            return "4";
        }
    }

    public static IdentifierEncoding getEncoding(String type, String name) {
        ArrayList<String> parts = IdentifierSpliter.splitIdentifier(name);
        assert parts != null;
        int numOfParts = parts.size();
        ArrayList<String> partNames = new ArrayList<>();
        ArrayList<String> partTypes = new ArrayList<>();
        ArrayList<Boolean> isInDics = new ArrayList<>();

        for (String part : parts) {
            partNames.add(part);
            partTypes.add(getPartType(part));
            isInDics.add(Dic.isInDict(part));
        }

        boolean containNum = Util.contains(name, "0123456789");
        boolean containUnderscore = Util.contains(name, "_");
        boolean containDollar = Util.contains(name, "$");
        return new IdentifierEncoding(name, type, numOfParts, partNames, partTypes, isInDics, containNum, containUnderscore, containDollar);
    }


    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("{");
        sb.append("\"name\":\"")
                .append(name).append('\"');
        sb.append(",\"type\":\"")
                .append(type).append('\"');
        sb.append(",\"numOfParts\":")
                .append(numOfParts);
        sb.append(",\"partNames\":")
                .append(Util.arrayListStringToString(partNames));
        sb.append(",\"partTypes\":")
                .append(Util.arrayListStringToString(partTypes));
        sb.append(",\"isInDics\":")
                .append(isInDics);
        sb.append(",\"containNum\":")
                .append(containNum);
        sb.append(",\"containUnderscore\":")
                .append(containUnderscore);
        sb.append(",\"containDollar\":")
                .append(containDollar);
        sb.append('}');
        return sb.toString();
    }
}
