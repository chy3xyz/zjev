const std = @import("std");
const alloc = @import("../core/alloc.zig");
const schema = @import("../core/schema.zig");

pub const CmpOp = enum { eq, ne, gt, lt, ge, le };
pub const Field = enum { value, confidence, abstention };
pub const FieldRef = struct { decision: []const u8, field: Field };
pub const Literal = union(enum) { str: []const u8, num: f32, boolean: bool };
pub const Operand = union(enum) { field: FieldRef, lit: Literal };
pub const Cmp = struct { op: CmpOp, lhs: Operand, rhs: Operand };
pub const Cond = union(enum) { cmp: Cmp, or_: []const Cond, and_: []const Cond, not: *const Cond, truthy: Operand };

pub const ParseError = error{ Syntax, TypeMismatch, UnknownField, UnknownDecision, RankNotComparable, OutOfMemory };

const fixture_schemas = [_]schema.DecisionSchema{
    .{ .choice = .{ .id = "risk", .options = &.{ "low", "high" }, .abstain = true } },
    .{ .noul = .{ .id = "flag", .abstain = true } },
    .{ .score = .{ .id = "sev", .scale = .{ .int = .{ .min = 1, .max = 5 } }, .abstain = false } },
    .{ .rank = .{ .id = "ord", .items = &.{ "x", "y" } } },
};

fn parseOk(a: alloc.Allocator, src: []const u8) !Cond {
    return parse(a, src, &fixture_schemas);
}

test "parse bare comparison with bare ident literal" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try parseOk(a, "risk == high");
    try std.testing.expect(c == .cmp);
    try std.testing.expect(c.cmp.op == .eq);
    try std.testing.expect(c.cmp.lhs == .field);
    try std.testing.expectEqualStrings("risk", c.cmp.lhs.field.decision);
    try std.testing.expect(c.cmp.lhs.field.field == .value);
    try std.testing.expect(c.cmp.rhs == .lit);
    try std.testing.expectEqualStrings("high", c.cmp.rhs.lit.str);
}

test "parse explicit value field with quoted string" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try parseOk(a, "risk.value == \"high\"");
    try std.testing.expect(c.cmp.op == .eq);
}

test "parse numeric comparison on score" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try parseOk(a, "sev >= 3");
    try std.testing.expect(c.cmp.op == .ge);
    try std.testing.expect(c.cmp.rhs.lit.num == 3.0);
}

test "parse confidence and abstention fields" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try parseOk(a, "risk.confidence > 0.5 and risk.abstention <= 0.1");
    try std.testing.expect(c == .and_);
    try std.testing.expect(c.and_.len == 2);
}

test "parse or and not" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try parseOk(a, "not risk == high or flag == true");
    try std.testing.expect(c == .or_);
    try std.testing.expect(c.or_[0] == .not);
}

test "parse parenthesized grouping" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try parseOk(a, "(risk == low or risk == high) and sev < 4");
    try std.testing.expect(c == .and_);
}

test "parse bare noul truthy" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try parseOk(a, "flag");
    try std.testing.expect(c == .truthy);
}

test "parse field vs field comparison" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try parseOk(a, "sev.confidence > risk.confidence");
    try std.testing.expect(c.cmp.lhs == .field);
    try std.testing.expect(c.cmp.rhs == .field);
}

test "reject rank value comparison" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectError(error.RankNotComparable, parse(arena.allocator(), "ord == x", &fixture_schemas));
}

test "reject bare ident literal on score" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectError(error.TypeMismatch, parse(arena.allocator(), "sev == high", &fixture_schemas));
}

test "reject gt on choice value" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectError(error.TypeMismatch, parse(arena.allocator(), "risk > low", &fixture_schemas));
}

test "reject boolean literal on choice" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectError(error.TypeMismatch, parse(arena.allocator(), "risk == true", &fixture_schemas));
}

test "reject number literal on choice" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectError(error.TypeMismatch, parse(arena.allocator(), "risk == 3", &fixture_schemas));
}

test "reject unknown decision" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectError(error.UnknownDecision, parse(arena.allocator(), "ghost == high", &fixture_schemas));
}

test "reject abstention on schema without abstain" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectError(error.TypeMismatch, parse(arena.allocator(), "sev.abstention > 0.1", &fixture_schemas));
}

test "reject syntax error" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectError(error.Syntax, parse(arena.allocator(), "risk ==", &fixture_schemas));
}

test "reject trailing garbage" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectError(error.Syntax, parse(arena.allocator(), "risk == high high", &fixture_schemas));
}

pub const ValCat = enum { str, boolean, num };

const Tok = union(enum) {
    ident: []const u8,
    str: []const u8,
    num: f32,
    eq,
    ne,
    gt,
    lt,
    ge,
    le,
    lparen,
    rparen,
    dot,
    eof,
};

fn lex(src: []const u8, toks: *std.ArrayList(Tok), a: alloc.Allocator) ParseError!void {
    var i: usize = 0;
    while (i < src.len) {
        const ch = src[i];
        if (ch == ' ' or ch == '\t' or ch == '\n' or ch == '\r') {
            i += 1;
            continue;
        }
        if (ch == '"') {
            const start = i + 1;
            var j = start;
            while (j < src.len and src[j] != '"') j += 1;
            if (j >= src.len) return error.Syntax;
            try toks.append(a, .{ .str = src[start..j] });
            i = j + 1;
            continue;
        }
        if (std.ascii.isDigit(ch) or (ch == '-' and i + 1 < src.len and std.ascii.isDigit(src[i + 1]))) {
            const start = i;
            i += 1;
            while (i < src.len and (std.ascii.isDigit(src[i]) or src[i] == '.')) i += 1;
            const v = std.fmt.parseFloat(f32, src[start..i]) catch return error.Syntax;
            try toks.append(a, .{ .num = v });
            continue;
        }
        if (std.ascii.isAlphabetic(ch) or ch == '_') {
            const start = i;
            while (i < src.len and (std.ascii.isAlphanumeric(src[i]) or src[i] == '_')) i += 1;
            try toks.append(a, .{ .ident = src[start..i] });
            continue;
        }
        switch (ch) {
            '=' => {
                if (i + 1 < src.len and src[i + 1] == '=') {
                    try toks.append(a, .eq);
                    i += 2;
                } else return error.Syntax;
            },
            '!' => {
                if (i + 1 < src.len and src[i + 1] == '=') {
                    try toks.append(a, .ne);
                    i += 2;
                } else return error.Syntax;
            },
            '>' => {
                if (i + 1 < src.len and src[i + 1] == '=') {
                    try toks.append(a, .ge);
                    i += 2;
                } else {
                    try toks.append(a, .gt);
                    i += 1;
                }
            },
            '<' => {
                if (i + 1 < src.len and src[i + 1] == '=') {
                    try toks.append(a, .le);
                    i += 2;
                } else {
                    try toks.append(a, .lt);
                    i += 1;
                }
            },
            '(' => {
                try toks.append(a, .lparen);
                i += 1;
            },
            ')' => {
                try toks.append(a, .rparen);
                i += 1;
            },
            '.' => {
                try toks.append(a, .dot);
                i += 1;
            },
            else => return error.Syntax,
        }
    }
    try toks.append(a, .eof);
}

const Parser = struct {
    a: alloc.Allocator,
    toks: []const Tok,
    pos: usize = 0,
    schemas: []const schema.DecisionSchema,

    fn peek(p: *Parser) Tok {
        return p.toks[p.pos];
    }

    fn take(p: *Parser) Tok {
        const t = p.toks[p.pos];
        if (t != .eof) p.pos += 1;
        return t;
    }

    fn isIdent(p: *Parser, word: []const u8) bool {
        const t = p.peek();
        return t == .ident and std.mem.eql(u8, t.ident, word);
    }

    fn parseOr(p: *Parser) ParseError!Cond {
        var parts: std.ArrayList(Cond) = .empty;
        try parts.append(p.a, try p.parseAnd());
        while (p.isIdent("or")) {
            _ = p.take();
            try parts.append(p.a, try p.parseAnd());
        }
        if (parts.items.len == 1) return parts.items[0];
        return .{ .or_ = try parts.toOwnedSlice(p.a) };
    }

    fn parseAnd(p: *Parser) ParseError!Cond {
        var parts: std.ArrayList(Cond) = .empty;
        try parts.append(p.a, try p.parseNot());
        while (p.isIdent("and")) {
            _ = p.take();
            try parts.append(p.a, try p.parseNot());
        }
        if (parts.items.len == 1) return parts.items[0];
        return .{ .and_ = try parts.toOwnedSlice(p.a) };
    }

    fn parseNot(p: *Parser) ParseError!Cond {
        if (p.isIdent("not")) {
            _ = p.take();
            const inner = try p.a.create(Cond);
            inner.* = try p.parseNot();
            return .{ .not = inner };
        }
        return p.parsePrimary();
    }

    fn parsePrimary(p: *Parser) ParseError!Cond {
        const t = p.peek();
        if (t == .lparen) {
            _ = p.take();
            const inner = try p.parseOr();
            if (p.take() != .rparen) return error.Syntax;
            return inner;
        }
        return p.parseCmp();
    }

    fn parseCmp(p: *Parser) ParseError!Cond {
        const lhs = try p.parseOperand();
        const t = p.peek();
        const op: CmpOp = switch (t) {
            .eq => .eq,
            .ne => .ne,
            .gt => .gt,
            .lt => .lt,
            .ge => .ge,
            .le => .le,
            else => {
                try checkTruthy(p.schemas, lhs);
                return .{ .truthy = lhs };
            },
        };
        _ = p.take();
        const rhs = try p.parseOperandRhs();
        try checkCmp(p.schemas, lhs, op, rhs);
        return .{ .cmp = .{ .op = op, .lhs = lhs, .rhs = rhs } };
    }

    // 右操作数：ident + "." + ident → 字段引用；裸 ident → option 名字面量
    // （SPEC §2.2 词法消歧规则）
    fn parseOperandRhs(p: *Parser) ParseError!Operand {
        const t = p.take();
        switch (t) {
            .str => |s| return .{ .lit = .{ .str = s } },
            .num => |n| return .{ .lit = .{ .num = n } },
            .ident => |id| {
                if (std.mem.eql(u8, id, "true")) return .{ .lit = .{ .boolean = true } };
                if (std.mem.eql(u8, id, "false")) return .{ .lit = .{ .boolean = false } };
                if (p.peek() == .dot) {
                    _ = p.take();
                    const f = p.take();
                    if (f != .ident) return error.Syntax;
                    const field: Field = if (std.mem.eql(u8, f.ident, "value"))
                        .value
                    else if (std.mem.eql(u8, f.ident, "confidence"))
                        .confidence
                    else if (std.mem.eql(u8, f.ident, "abstention"))
                        .abstention
                    else
                        return error.UnknownField;
                    return .{ .field = .{ .decision = id, .field = field } };
                }
                return .{ .lit = .{ .str = id } };
            },
            else => return error.Syntax,
        }
    }

    // 左操作数：一律按字段引用解析（SPEC §2.2）
    fn parseOperand(p: *Parser) ParseError!Operand {
        const t = p.take();
        switch (t) {
            .str => |s| return .{ .lit = .{ .str = s } },
            .num => |n| return .{ .lit = .{ .num = n } },
            .ident => |id| {
                if (std.mem.eql(u8, id, "true")) return .{ .lit = .{ .boolean = true } };
                if (std.mem.eql(u8, id, "false")) return .{ .lit = .{ .boolean = false } };
                if (p.peek() == .dot) {
                    _ = p.take();
                    const f = p.take();
                    if (f != .ident) return error.Syntax;
                    const field: Field = if (std.mem.eql(u8, f.ident, "value"))
                        .value
                    else if (std.mem.eql(u8, f.ident, "confidence"))
                        .confidence
                    else if (std.mem.eql(u8, f.ident, "abstention"))
                        .abstention
                    else
                        return error.UnknownField;
                    return .{ .field = .{ .decision = id, .field = field } };
                }
                return .{ .field = .{ .decision = id, .field = .value } };
            },
            else => return error.Syntax,
        }
    }
};

fn schemaById(schemas: []const schema.DecisionSchema, id: []const u8) ?schema.DecisionSchema {
    for (schemas) |sc| {
        if (std.mem.eql(u8, sc.id(), id)) return sc;
    }
    return null;
}

pub fn category(fr: FieldRef, schemas: []const schema.DecisionSchema) ParseError!ValCat {
    const sc = schemaById(schemas, fr.decision) orelse return error.UnknownDecision;
    switch (fr.field) {
        .confidence => return .num,
        .abstention => {
            const has = switch (sc) {
                .choice => |c| c.abstain,
                .noul => |n| n.abstain,
                .score => |s| s.abstain,
                .rank => false,
            };
            if (!has) return error.TypeMismatch;
            return .num;
        },
        .value => return switch (sc) {
            .choice => .str,
            .noul => .boolean,
            .score => .num,
            .rank => error.RankNotComparable,
        },
    }
}

fn operandCat(schemas: []const schema.DecisionSchema, o: Operand) ParseError!ValCat {
    return switch (o) {
        .field => |fr| category(fr, schemas),
        .lit => |l| switch (l) {
            .str => .str,
            .num => .num,
            .boolean => .boolean,
        },
    };
}

fn checkCmp(schemas: []const schema.DecisionSchema, lhs: Operand, op: CmpOp, rhs: Operand) ParseError!void {
    if (lhs != .field) return error.Syntax;
    const lc = try category(lhs.field, schemas);
    const rc = try operandCat(schemas, rhs);
    if (lc != rc) return error.TypeMismatch;
    if (lc == .str and op != .eq and op != .ne) return error.TypeMismatch;
    if (lc == .boolean and op != .eq and op != .ne) return error.TypeMismatch;
}

fn checkTruthy(schemas: []const schema.DecisionSchema, o: Operand) ParseError!void {
    if (o != .field) return error.Syntax;
    if (try category(o.field, schemas) != .boolean) return error.TypeMismatch;
}

pub fn parse(a: alloc.Allocator, src: []const u8, schemas: []const schema.DecisionSchema) ParseError!Cond {
    if (src.len > 1024) return error.Syntax;
    var toks: std.ArrayList(Tok) = .empty;
    defer toks.deinit(a);
    try lex(src, &toks, a);
    var p = Parser{ .a = a, .toks = toks.items, .schemas = schemas };
    const c = try p.parseOr();
    if (p.peek() != .eof) return error.Syntax;
    return c;
}

// ---- eval（Task 3）----

const result = @import("../core/result.zig");

pub const EvalError = error{ UnknownDecision, ConditionDependency, TypeMismatch };

fn drChoice(id: []const u8, v: []const u8, conf: f32, abst: ?f32) result.DecisionResult {
    return .{
        .id = id,
        .type = .choice,
        .value = .{ .choice = v },
        .uncertainty = .{ .confidence = conf, .abstention = abst },
    };
}

fn drNoul(id: []const u8, b: bool, pyes: f32) result.DecisionResult {
    return .{
        .id = id,
        .type = .noul,
        .value = .{ .noul = b },
        .probability = pyes,
        .uncertainty = .{ .confidence = pyes },
    };
}

test "eval eq true on choice value" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const rs = [_]result.DecisionResult{drChoice("risk", "high", 0.8, 0.1)};
    try std.testing.expect(try eval(try parse(a, "risk == high", &fixture_schemas), &rs));
    try std.testing.expect(!(try eval(try parse(a, "risk == low", &fixture_schemas), &rs)));
}

test "eval numeric on score value" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r: result.DecisionResult = .{
        .id = "sev",
        .type = .score,
        .value = .{ .score = 3.6 },
        .uncertainty = .{ .confidence = 0.5 },
    };
    const rs = [_]result.DecisionResult{r};
    try std.testing.expect(try eval(try parse(a, "sev >= 3", &fixture_schemas), &rs));
    try std.testing.expect(!(try eval(try parse(a, "sev < 3", &fixture_schemas), &rs)));
}

test "eval and or not" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const rs = [_]result.DecisionResult{
        drChoice("risk", "high", 0.8, 0.1),
        drNoul("flag", true, 0.7),
    };
    try std.testing.expect(try eval(try parse(a, "risk == high and flag", &fixture_schemas), &rs));
    try std.testing.expect(try eval(try parse(a, "risk == low or flag", &fixture_schemas), &rs));
    try std.testing.expect(!(try eval(try parse(a, "not flag", &fixture_schemas), &rs)));
}

test "eval confidence and abstention" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const rs = [_]result.DecisionResult{drChoice("risk", "high", 0.8, 0.1)};
    try std.testing.expect(try eval(try parse(a, "risk.confidence > 0.5 and risk.abstention <= 0.1", &fixture_schemas), &rs));
}

test "eval noul boolean literal" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const rs = [_]result.DecisionResult{drNoul("flag", false, 0.3)};
    try std.testing.expect(try eval(try parse(a, "flag == false", &fixture_schemas), &rs));
    try std.testing.expect(!(try eval(try parse(a, "flag", &fixture_schemas), &rs)));
}

test "eval errors when decision not executed" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try parse(a, "risk == high", &fixture_schemas);
    try std.testing.expectError(error.ConditionDependency, eval(c, &.{}));
}

const V = union(enum) { str: []const u8, num: f32, boolean: bool };

fn findResult(results: []const result.DecisionResult, id: []const u8) ?result.DecisionResult {
    for (results) |r| {
        if (std.mem.eql(u8, r.id, id)) return r;
    }
    return null;
}

fn fieldVal(fr: FieldRef, results: []const result.DecisionResult) EvalError!V {
    const r = findResult(results, fr.decision) orelse return error.ConditionDependency;
    switch (fr.field) {
        .confidence => return .{ .num = r.uncertainty.confidence },
        .abstention => {
            const ab = r.uncertainty.abstention orelse return error.TypeMismatch;
            return .{ .num = ab };
        },
        .value => return switch (r.value) {
            .choice => |s| .{ .str = s },
            .noul => |b| .{ .boolean = b },
            .score => |x| .{ .num = x },
            .rank => return error.TypeMismatch,
        },
    }
}

fn operandVal(o: Operand, results: []const result.DecisionResult) EvalError!V {
    return switch (o) {
        .field => |fr| fieldVal(fr, results),
        .lit => |l| switch (l) {
            .str => |s| .{ .str = s },
            .num => |n| .{ .num = n },
            .boolean => |b| .{ .boolean = b },
        },
    };
}

fn cmpVals(op: CmpOp, x: V, y: V) EvalError!bool {
    if (std.meta.activeTag(x) != std.meta.activeTag(y)) return error.TypeMismatch;
    return switch (x) {
        .str => |s| switch (op) {
            .eq => std.mem.eql(u8, s, y.str),
            .ne => !std.mem.eql(u8, s, y.str),
            else => return error.TypeMismatch,
        },
        .boolean => |b| switch (op) {
            .eq => b == y.boolean,
            .ne => b != y.boolean,
            else => return error.TypeMismatch,
        },
        .num => |u| switch (op) {
            .eq => u == y.num,
            .ne => u != y.num,
            .gt => u > y.num,
            .lt => u < y.num,
            .ge => u >= y.num,
            .le => u <= y.num,
        },
    };
}

pub fn eval(c: Cond, results: []const result.DecisionResult) EvalError!bool {
    return switch (c) {
        .cmp => |k| cmpVals(k.op, try operandVal(k.lhs, results), try operandVal(k.rhs, results)),
        .or_ => |parts| blk: {
            for (parts) |p| {
                if (try eval(p, results)) break :blk true;
            }
            break :blk false;
        },
        .and_ => |parts| blk: {
            for (parts) |p| {
                if (!(try eval(p, results))) break :blk false;
            }
            break :blk true;
        },
        .not => |inner| !(try eval(inner.*, results)),
        .truthy => |o| switch (try operandVal(o, results)) {
            .boolean => |b| b,
            else => return error.TypeMismatch,
        },
    };
}
