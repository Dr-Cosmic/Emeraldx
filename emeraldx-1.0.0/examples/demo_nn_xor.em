# A neural network learns XOR - the classic nonlinear test.
import nn;
import prng;
import strx;
int acc = 0;
int b = 0;
int ep = 0;
int fw = 0;
int g = 0;
int gb = 0;
int gw = 0;
int i = 0;
int loss = 0;
int net = 0;
int o = 0;
int opt = 0;
int sc = 0;
int st = 0;
int xs = 0;
int ys = 0;
int z = 0;
# emx-scope-safe

print("== emeraldx: neural network (XOR) ==");
xs = [[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]];
ys = [[0.0],[1.0],[1.0],[0.0]];
b = nn.build([2, 6, 1], ["tanh", "sigmoid"], prng.seed_from(11));
net = b[0];
opt = nn.opt_adam(net, 0.05);
for (ep = 0; ep < 400; ++ep) {
    z = nn.zero_grads(net);
    gw = z[0]; gb = z[1];
    loss = 0.0;
    for (i = 0; i < 4; ++i) {
        fw = nn.forward(net, xs[i]);
        o = nn.output(fw);
        e = o[0] - ys[i][0];
        loss = loss + e * e;
        g = nn.backward(net, fw, [2.0 * e]);
        acc = nn.add_grads(gw, gb, g[0], g[1]);
        gw = acc[0]; gb = acc[1];
    }
    sc = nn.scale_grads(gw, gb, 0.25);
    st = nn.apply_grads(net, opt, sc[0], sc[1]);
    net = st[0]; opt = st[1];
    if (ep == 0 || ep == 399) { print("epoch ", ep, " loss ", strx.fmt_f(loss / 4.0, 6)); }
}
for (i = 0; i < 4; ++i) {
    fw = nn.forward(net, xs[i]);
    o = nn.output(fw);
    print("  ", xs[i], " -> ", strx.fmt_f(o[0], 3));
}
