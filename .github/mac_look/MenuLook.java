import java.awt.Component;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Frame;
import java.awt.Rectangle;
import java.awt.Robot;
import java.awt.Toolkit;
import java.awt.Window;
import java.awt.event.MouseEvent;
import java.io.File;
import java.lang.instrument.Instrumentation;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import javax.imageio.ImageIO;
import javax.swing.AbstractButton;
import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JSeparator;
import javax.swing.SwingUtilities;
import javax.swing.UIManager;

/**
 * TEST ONLY (the Mac workflow): the game client as a person finds it - its AI menu and the AI panel's strength knob.
 *
 * A java agent (-javaagent:MenuLook.jar=FOLDER through JAVA_TOOL_OPTIONS) in the REAL client, started as a person
 * starts it (Play_vs_AI.command's wotr_bridge/play.py).  It waits for the game's window and its "AI" menu and writes
 * to FOLDER/look.json:
 *   the AI menu's entries, and that they are entries only (on a Mac the menus are drawn by macOS in the bar at the
 *     top of the screen, which shows nothing else);
 *   the strength knob (the component named "ai.strength") of the AI panel: showing, where, the words over it -
 *     and the simulations knob under it ("ai.sims"): showing, its words;
 *   the knob pressed and let go on another position, by mouse events put into the client's own event queue: the
 *     file play_settings.json beside the client must then state that position.
 * It keeps two pictures of the screen (screen_robot.png by Java, screen_capture.png by macOS) and ends the client:
 * 0 = looked, 3 = no AI menu came, 4 = the look itself failed.
 * It knows two things of the client: the menu's title and the knob's name; the knob's stops are read from its own
 * fields (min, max, value, pad()).
 */
public class MenuLook {
    static final String KNOB = "ai.strength";
    static final String SIMS_KNOB = "ai.sims";
    static final int WAIT_S = 240;
    static File dir;
    static final StringBuilder json = new StringBuilder("{\n");

    public static void premain(String args, Instrumentation inst) {
        dir = new File(args);
        Thread t = new Thread(new Runnable() { public void run() { look(); } }, "MenuLook");
        t.setDaemon(true);
        t.start();
    }

    static String q(String s) {
        if (s == null) return "null";
        return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", " ") + "\"";
    }

    static void put(String key, String rawValue) {
        json.append(" ").append(q(key)).append(": ").append(rawValue).append(",\n");
    }

    static void finish(int code) {
        put("exit", String.valueOf(code));
        String text = json.toString();
        text = text.substring(0, text.length() - 2) + "\n}\n";
        try {
            Files.write(new File(dir, "look.json").toPath(), text.getBytes(StandardCharsets.UTF_8));
        } catch (Throwable t) {
            t.printStackTrace();
        }
        Runtime.getRuntime().halt(code);
    }

    static JMenu aiMenu() {
        Frame[] frames = Frame.getFrames();
        for (int i = 0; i < frames.length; i++) {
            if (!(frames[i] instanceof JFrame) || !frames[i].isShowing()) continue;
            JMenuBar bar = ((JFrame) frames[i]).getJMenuBar();
            if (bar == null) continue;
            for (int k = 0; k < bar.getMenuCount(); k++) {
                JMenu m = bar.getMenu(k);
                if (m != null && "AI".equals(m.getText())) return m;
            }
        }
        return null;
    }

    static Component named(Component c, String name) {
        if (name.equals(c.getName())) return c;
        if (c instanceof Container) {
            Component[] kids = ((Container) c).getComponents();
            for (int i = 0; i < kids.length; i++) {
                Component hit = named(kids[i], name);
                if (hit != null) return hit;
            }
        }
        return null;
    }

    static Component knob() {
        Window[] all = Window.getWindows();
        for (int i = 0; i < all.length; i++) {
            if (!all[i].isShowing()) continue;
            Component hit = named(all[i], KNOB);
            if (hit != null) return hit;
        }
        return null;
    }

    static Field field(Class<?> c, String name) throws Exception {
        for (; c != null; c = c.getSuperclass()) {
            try {
                Field f = c.getDeclaredField(name);
                f.setAccessible(true);
                return f;
            } catch (NoSuchFieldException e) { /* the class above */ }
        }
        throw new NoSuchFieldException(name);
    }

    static Object call(Object o, String name) throws Exception {
        for (Class<?> c = o.getClass(); c != null; c = c.getSuperclass()) {
            try {
                Method m = c.getDeclaredMethod(name);
                m.setAccessible(true);
                return m.invoke(o);
            } catch (NoSuchMethodException e) { /* the class above */ }
        }
        throw new NoSuchMethodException(name);
    }

    static String settings() {
        try {
            return new String(Files.readAllBytes(new File("play_settings.json").toPath()), StandardCharsets.UTF_8).trim();
        } catch (Throwable t) {
            return "(" + t + ")";
        }
    }

    static void pictures(String stem) {
        try {
            Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
            ImageIO.write(new Robot().createScreenCapture(new Rectangle(screen)), "png", new File(dir, stem + "_robot.png"));
        } catch (Throwable t) {
            put(stem + "_robot_error", q(String.valueOf(t)));
        }
        if (new File("/usr/sbin/screencapture").isFile()) {
            try {
                new ProcessBuilder("/usr/sbin/screencapture", "-x", new File(dir, stem + "_capture.png").getPath())
                        .inheritIO().start().waitFor();
            } catch (Throwable t) {
                put(stem + "_capture_error", q(String.valueOf(t)));
            }
        }
    }

    static void look() {
        try {
            Thread.sleep(15000);                                     // the client's own start first
            put("os", q(System.getProperty("os.name") + " " + System.getProperty("os.version") + " " + System.getProperty("os.arch")));
            put("java", q(System.getProperty("java.version") + " " + System.getProperty("java.vendor")));
            JMenu menu = null;
            for (int s = 15; s < WAIT_S && menu == null; s += 2) {
                menu = aiMenu();
                if (menu == null) Thread.sleep(2000);
            }
            put("look_and_feel", q(UIManager.getLookAndFeel().getClass().getName()));
            if (menu == null) {
                put("ai_menu", "false");
                pictures("screen");
                finish(3);
            }
            put("ai_menu", "true");
            final JMenu ai = menu;
            Thread.sleep(5000);                                      // the menu and the panel have been built
            final int[] target = new int[1];
            SwingUtilities.invokeAndWait(new Runnable() {
                public void run() {
                    try {
                        JMenuBar bar = (JMenuBar) ai.getParent();
                        put("menu_bar_in_window", String.valueOf(bar.isShowing() && bar.getHeight() > 0));
                        StringBuilder entries = new StringBuilder("[");
                        boolean only = true, legacy = false;
                        Component[] kids = ai.getMenuComponents();
                        for (int i = 0; i < kids.length; i++) {
                            Component c = kids[i];
                            only &= c instanceof JMenuItem || c instanceof JSeparator;
                            String words = c instanceof AbstractButton ? ((AbstractButton) c).getText() : c.getClass().getSimpleName();
                            legacy |= words != null && words.toLowerCase().contains("legacy");
                            entries.append(i == 0 ? "" : ", ").append(q(words));
                        }
                        put("entries", entries.append("]").toString());
                        put("menu_entries_only", String.valueOf(only));
                        put("legacy_entry", String.valueOf(legacy));
                        Component sims = null;
                        Window[] all = Window.getWindows();
                        for (int i = 0; i < all.length && sims == null; i++)
                            if (all[i].isShowing()) sims = named(all[i], SIMS_KNOB);
                        put("sims_knob_showing", String.valueOf(sims != null && sims.isShowing()));
                        if (sims != null) put("sims_knob_words", q(String.valueOf(call(sims.getParent(), "searchWords"))));
                        Component knob = knob();
                        put("knob_found", String.valueOf(knob != null));
                        put("knob_showing", String.valueOf(knob != null && knob.isShowing()));
                        if (knob == null || !knob.isShowing()) return;
                        put("knob_on_screen", q(String.valueOf(knob.getLocationOnScreen()) + " " + knob.getWidth() + "x" + knob.getHeight()));
                        put("knob_words", q(String.valueOf(call(knob.getParent(), "words"))));
                        int min = field(knob.getClass(), "min").getInt(knob), max = field(knob.getClass(), "max").getInt(knob);
                        int value = field(knob.getClass(), "value").getInt(knob);
                        int pad = ((Integer) call(knob, "pad")).intValue();
                        put("knob_stands_at", q(value + " of " + min + ".." + max));
                        put("settings_before", q(settings()));
                        target[0] = value == 2 ? 3 : 2;
                        int x = pad + (int) Math.round((target[0] - min) / (double) Math.max(1, max - min) * (knob.getWidth() - 2 * pad));
                        int y = knob.getHeight() / 2;
                        long t = System.currentTimeMillis();
                        knob.dispatchEvent(new MouseEvent(knob, MouseEvent.MOUSE_PRESSED, t, 0, x, y, 1, false, MouseEvent.BUTTON1));
                        knob.dispatchEvent(new MouseEvent(knob, MouseEvent.MOUSE_RELEASED, t, 0, x, y, 1, false, MouseEvent.BUTTON1));
                    } catch (Throwable t) {
                        put("look_error", q(String.valueOf(t)));
                    }
                }
            });
            Thread.sleep(1500);
            if (target[0] > 0) {
                String after = settings();
                put("knob_moved_to", String.valueOf(target[0]));
                put("settings_after", q(after));
                put("knob_writes_file", String.valueOf(after.replace(" ", "").contains("\"strength\":" + target[0])));
                SwingUtilities.invokeAndWait(new Runnable() {
                    public void run() {
                        try {
                            Component knob = knob();
                            if (knob != null) put("knob_words_after", q(String.valueOf(call(knob.getParent(), "words"))));
                        } catch (Throwable t) {
                            put("look_error_after", q(String.valueOf(t)));
                        }
                    }
                });
            }
            pictures("screen");
            finish(0);
        } catch (Throwable t) {
            put("error", q(String.valueOf(t)));
            finish(4);
        }
    }
}
