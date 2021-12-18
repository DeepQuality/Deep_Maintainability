package main.util;

import java.util.ArrayList;

public class IdentifierSpliter {
    public static boolean isNum(char c) {
        return c >= '0' && c <= '9';
    }

    public static boolean isLetter(char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    }

    // str only consists of letters
    public static ArrayList<String> splitBigLetter(String str) {
        String ori = str;
        char[] list = str.toCharArray();
        str = "A" + str + "A";
        char[] tempList = str.toCharArray();

        for (int i = 1; i < tempList.length - 1; i++) {
            if (tempList[i] >= 'A' && tempList[i] <= 'Z' &&
                    tempList[i - 1] >= 'A' && tempList[i - 1] <= 'Z' &&
                    tempList[i + 1] >= 'A' && tempList[i + 1] <= 'Z') {
                list[i - 1] = (char) (tempList[i] - 'A' + 'a');
            } else {
                list[i - 1] = tempList[i];
            }
        }

        str = new String(list);
        ArrayList<String> result = new ArrayList<>();
        int startPositionOfSubstring = 0;
        str = str + 'A';
        for (int endPositionOfSubstring = 0; endPositionOfSubstring < str.length(); endPositionOfSubstring++) {
            if (str.charAt(endPositionOfSubstring) >= 'A' && str.charAt(endPositionOfSubstring) <= 'Z') {
                // to exclude initial up case letter
                if (str.substring(startPositionOfSubstring, endPositionOfSubstring).length() > 0) {
                    // to lower case
                    result.add(ori.substring(startPositionOfSubstring, endPositionOfSubstring));
                    startPositionOfSubstring = endPositionOfSubstring;
                }
            }
        }
        return result;
    }

    // identifiers consist of
    // letters
    // numbers
    // _
    // $
    // not start with numbers
    // the size of return value may be 0 (e.g., $)
    public static ArrayList<String> splitIdentifier(String str) {
        ArrayList<String> result = new ArrayList<>();
        if (str == null || str.length() == 0) {
            System.out.println("error: split");
            return null;
        }
        // delete $, _, numbers at the beginning of str
        while (str.length() > 0 && (str.charAt(0) == '$' || str.charAt(0) == '_' || isNum(str.charAt(0)))) {
            str = str.substring(1);
        }
        if (str.length() == 0) return result;
        // replace $, _, numbers with the separator #
        str = str.replaceAll("\\d", "#");
        str = str.replaceAll("\\_", "#");
        str = str.replaceAll("\\$", "#");

        for (String string : str.split("#")) {
            if (string.length() > 0) {
                result.addAll(splitBigLetter(string));
            }
        }
        return result;
    }
}
