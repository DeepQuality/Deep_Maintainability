package main.util;

import java.io.*;
import java.util.ArrayList;

public class Util {

    public static String arrayListStringToString(ArrayList<String> strs) {
        if (strs.size() == 0) {
            return "[]";
        }
        StringBuilder stringBuilder = new StringBuilder("[");
        for (int i = 0; i < strs.size() - 1; i++) {
            stringBuilder.append("\"");
            stringBuilder.append(strs.get(i));
            stringBuilder.append("\",");
        }
        stringBuilder.append("\"");
        stringBuilder.append(strs.get(strs.size()-1));
        stringBuilder.append("\"]");
        return stringBuilder.toString();
    }

    public static boolean contains(String str, String obj) {
        char[] ori = str.toCharArray();
        char[] sub = obj.toCharArray();
        for (char value : ori) {
            for (char c : sub) {
                if (value == c) {
                    return true;
                }
            }
        }
        return false;
    }
}